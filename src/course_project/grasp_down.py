#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import numpy as np
import ikpy.chain
import xacro
import tempfile
import os
from collections import deque
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray 
import time
# ==========================================
# 🔧 统一配置区
# ==========================================
class SYSTEM_CONFIG:
    # 1. 坐标系偏置修正 (单位: 米)
    OFFSET_X =  0.00    
    OFFSET_Y =  0.00    
    OFFSET_Z =  -0.03    

    # 2. 安全避障参数 (关键修改)
    # 机械臂从归位状态恢复时，先去这个点，避开正前方的桌子
    # [X, Y, Z] -> 建议设置在机械臂侧面且较高位置
    # 例如: X=0.10(稍微向前), Y=0.25(偏左/右), Z=0.45(举高)
    SAFE_PARK_POS = [0.15, 0.2, 0.55] 
    SAFE_PARK_THRESHOLD = 0.05

    # 3. 抓取逻辑参数
    PRE_GRASP_HEIGHT = 0.03    # 预备高度
    GRASP_HEIGHT_OFFSET = 0.02 # 抓取微调 (接近物体表面的高度)
    LIFT_HEIGHT = 0.20         # 抓取后抬起的高度
    
    GRIPPER_FORCE = 3.0      # 夹爪闭合力度
    GRIPPER_OPEN = 0.0         # 夹爪张开力度

    # 4. 速度平滑参数
    # 每次控制周期(0.1s)末端移动的最大距离(米)
    # 0.015m / 0.1s = 0.15m/s (缓缓移动)
    MAX_STEP_SIZE = 0.015       

    # 5. 初始归位参数
    HOME_JOINTS = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0] 
    HOME_THRESHOLD = 0.05 
    FILTER_WINDOW_SIZE = 5      

# ==========================================

class DataProcessor:
    """数据处理模块：负责滤波和补偿"""
    def __init__(self):
        self.history = deque(maxlen=SYSTEM_CONFIG.FILTER_WINDOW_SIZE)

    def process(self, raw_point):
        self.history.append(raw_point)
        if not self.history:
            return raw_point
        avg_data = np.mean(np.array(self.history), axis=0)
        corrected_pos = [
            avg_data[0] + SYSTEM_CONFIG.OFFSET_X,
            avg_data[1] + SYSTEM_CONFIG.OFFSET_Y,
            avg_data[2] + SYSTEM_CONFIG.OFFSET_Z
        ]
        return corrected_pos

class DebugGraspController(Node):
    def __init__(self):
        super().__init__('debug_grasp_controller')
        
        self.processor = DataProcessor()
        self.latched_target = None
        
        # 预先定义变量，防止初始化失败导致崩溃
        self.ik_chain = None
        self.active_mask = []
        
        # 初始化模型
        self.init_ik_chain()
        
        self.current_joints = None
        self.last_target_pos = None # 用于插值平滑
        
        # 订阅与发布
        self.joint_sub = self.create_subscription(JointState, '/joint_states', self.joint_cb, 10)
        self.vision_sub = self.create_subscription(Float64MultiArray, '/openarm/target_object', self.vision_cb, 10)
        self.cmd_pub = self.create_publisher(Float64MultiArray, '/openarm/joint_commands', 10)
        
        # 状态机定义
        # -1: 归位 (Homing) -> 关节直接控制
        #  0: 安全抬起 (Safe Park) -> 避开桌子，移动到侧上方
        #  1: 视觉搜索 (Search) -> 悬停在高处等待视觉锁定
        #  2: 移动到预备点 (Approach) -> 下降到物体上方
        #  3: 下放 (Down) -> 缓慢接触
        #  4: 抓取 (Grasp) -> 闭合
        #  5: 抬起 (Lift) -> 举起
        self.state = 0
        
        # 提高频率到 10Hz 以获得丝滑插值
        self.timer = self.create_timer(0.1, self.control_loop) 
        self.get_logger().info("🚀 抓取控制器启动！等待关节数据...")

    def init_ik_chain(self):
        """修复后的 IK 链初始化函数"""
        xacro_file = "/home/zheng/openarm_ws/src/openarm_description/urdf/robot/v10.urdf.xacro"
        
        try:
            # 1. 处理 Xacro
            doc = xacro.process_file(xacro_file, mappings={'bimanual': 'true'})
            
            # 2. 写入临时文件 (修复：使用 mode='w' 且直接写入字符串)
            with tempfile.NamedTemporaryFile(suffix='.urdf', mode='w', delete=False) as tmp:
                tmp.write(doc.toprettyxml(indent='  '))
                urdf_path = tmp.name

            # 3. 加载 IK Chain
            self.ik_chain = ikpy.chain.Chain.from_urdf_file(urdf_path, base_elements=["world"])
            
            # 4. 设置 Active Mask (仅控制左臂)
            self.active_mask = [False] * len(self.ik_chain.links)
            for i, link in enumerate(self.ik_chain.links):
                if "openarm_left_joint" in link.name: 
                    self.active_mask[i] = True
            self.ik_chain.active_links_mask = self.active_mask
            
            # 5. 清理
            os.unlink(urdf_path)
            self.get_logger().info("✅ IK 链初始化成功")
            
        except Exception as e:
            self.get_logger().error(f"❌ 模型初始化失败: {e}")
            self.ik_chain = None # 标记为失败

    def joint_cb(self, msg):
        try:
            # 根据你的实际关节名称调整索引
            left_arm_indices = [msg.name.index(f'openarm_left_joint{i}') for i in range(1, 8)]
            self.current_joints = [msg.position[i] for i in left_arm_indices]
        except ValueError: 
            pass

    def vision_cb(self, msg):
        # ⚠️ 只有在 State 1 (搜索状态) 下才响应视觉
        # 必须等待机械臂先移动到安全高点(State 0完成)
        if self.state != 1: return

        data = msg.data
        if len(data) < 3: return
        x, y, z = data[0], data[1], data[2]

        if abs(x) < 0.001 and abs(y) < 0.001: return

        final_pos = self.processor.process([x, y, z])

        # 锁定目标
        self.latched_target = final_pos
        if len(self.processor.history) >= SYSTEM_CONFIG.FILTER_WINDOW_SIZE:
             self.get_logger().info(f"🎯 视觉锁定目标: {np.round(final_pos, 3)}")
             self.state = 1.5 # 跳转到预备状态
             self.pre_open_time=time.time()
    def get_vertical_down_orientation(self):
        """生成垂直向下的旋转矩阵 (末端 Z 轴朝向 World -Z)"""
        # 常见定义：X轴向前，Z轴向下
        # 如果你的机械臂手抓方向反了，请微调这个矩阵
        return np.array([
            [1,  0,  0],
            [0, -1,  0],
            [0,  0, -1] 
        ])
    def get_horizontal_forward_orientation(self):
    # 末端执行器的 X 轴向右，Z 轴水平指向目标
        return np.array([
        [0,  0, 1],   # X轴不变，Z轴水平指向目标
        [0,  -1,  0],   # Y轴不变
        [1,  0,  0]    # Z轴与目标位置对齐
    ])
    def smooth_move_to(self, current, target, step_size):
        """线性插值平滑器"""
        diff = target - current
        dist = np.linalg.norm(diff)
        
        if dist <= step_size:
            return target
        else:
            direction = diff / dist
            return current + direction * step_size

    def control_loop(self):
        # 1. 安全检查：如果 IK 没初始化，不运行
        if self.ik_chain is None:
            if self.state != -99:
                self.get_logger().error("⚠️ IK 未初始化，等待修复...")
                self.state = -99
            return

        if self.current_joints is None: return

        # 2. 准备 IK 计算所需的关节数据
        full_joints = [0.0] * len(self.ik_chain.links)
        idx = 0
        for i, active in enumerate(self.active_mask):
            if active and idx < len(self.current_joints):
                val = self.current_joints[idx]
                
                # === [新增] 核心修复：输入数据钳制 ===
                # 获取该关节在 URDF 中的物理限制
                lower = self.ik_chain.links[i].bounds[0]
                upper = self.ik_chain.links[i].bounds[1]
                
                # 如果 MuJoCo 发过来的数据稍微越界了(例如 3.14159)，强行拉回(3.14)
                # 加上 0.001 的余量防止边界浮点数误差
                #if val < lower: val = lower + 0.001
                if val > upper: val = upper - 0.001
                
                full_joints[i] = val
                # ==================================
                
                idx += 1
        
        # FK 获取当前末端位置
        current_pose_matrix = self.ik_chain.forward_kinematics(full_joints)
        current_ee_pos = current_pose_matrix[:3, 3]

        # 初始化插值器起点
        if self.last_target_pos is None:
            self.last_target_pos = current_ee_pos

        # 默认目标
        final_target_pos = current_ee_pos 
        gripper_cmd = SYSTEM_CONFIG.GRIPPER_OPEN
        
        # 强制姿态控制：全程垂直向下
        target_orientation = self.get_horizontal_forward_orientation()
        orientation_mode = "all" 

        # ===========================
        # 🤖 状态机逻辑
        # ===========================
        
        if self.state == -1: # === 归位 (Homing) ===
            # 直接发送关节角度，不走 IK
            msg = Float64MultiArray()
            cmd_joints = SYSTEM_CONFIG.HOME_JOINTS[:len(self.current_joints)]
            msg.data = cmd_joints + [SYSTEM_CONFIG.GRIPPER_OPEN]
            self.cmd_pub.publish(msg)
            
            # 检查误差
            curr = np.array(self.current_joints)
            tgt = np.array(cmd_joints)
            if np.linalg.norm(curr - tgt) < SYSTEM_CONFIG.HOME_THRESHOLD:
                self.get_logger().info("✅ 归位完成，准备前往安全高点...")
                self.processor.history.clear()
                self.last_target_pos = current_ee_pos # 重置插值器
                self.state = 0 # -> 跳转到安全抬起
            return # 归位时不执行后续 IK

        elif self.state == 0: # === 安全抬起 (Safe Park) ===
            # 关键步骤：先走到侧面上方，绕过桌子
            final_target_pos = np.array(SYSTEM_CONFIG.SAFE_PARK_POS)
            
            # 判断是否到达
            if np.linalg.norm(current_ee_pos - final_target_pos) < SYSTEM_CONFIG.SAFE_PARK_THRESHOLD:
                self.get_logger().info("🛡️ 已到达安全高点，开始视觉搜索...")
                self.state = 1 # -> 跳转到搜索
                
                
        elif self.state == 1.5:  # === Pre-Open ===
            final_target_pos = np.array(self.latched_target)
            final_target_pos[2] += SYSTEM_CONFIG.PRE_GRASP_HEIGHT

    # 强制张开夹爪
            gripper_cmd = SYSTEM_CONFIG.GRIPPER_OPEN

    # 保持 0.8 秒
            if time.time() - self.pre_open_time > 3:
                self.get_logger().info("⬇️ 夹爪已完全张开，开始下放")
                self.state = 2

        elif self.state == 1: # === 视觉搜索 (Search) ===
            # 保持在安全高点不动，等待 Vision Callback 触发
            final_target_pos = np.array(SYSTEM_CONFIG.SAFE_PARK_POS)
            # 注意：一旦 vision_cb 锁定目标，它会将 self.state 改为 2


       

        elif self.state == 5: # === 抬起 (Lift) ===
            target = list(self.latched_target)
            target[2] += SYSTEM_CONFIG.LIFT_HEIGHT
            final_target_pos = np.array(target)
            gripper_cmd = SYSTEM_CONFIG.GRIPPER_FORCE
        elif self.state == 4: # === 抓取 (Grasp) ===
            target = list(self.latched_target)
            target[2] += SYSTEM_CONFIG.GRASP_HEIGHT_OFFSET
            final_target_pos = np.array(target)
            gripper_cmd = SYSTEM_CONFIG.GRIPPER_FORCE
            
            # 为了简单，直接切换；建议实际使用中加个 timer 延时1秒等待闭合
            self.state = 5 
        elif self.state == 3: # === 下放 (Down) ===
            target = list(self.latched_target)
            target[0] += SYSTEM_CONFIG.GRASP_HEIGHT_OFFSET
            final_target_pos = np.array(target)
            gripper_cmd = SYSTEM_CONFIG.GRIPPER_OPEN  # 再保险一次

            if np.linalg.norm(current_ee_pos - final_target_pos) < 0.01:
                self.get_logger().info("👌 到达抓取位")
                self.state = 4
        elif self.state == 2: # === 接近 (Approach) ===
            # 从高点向下移动到物体上方
            target = list(self.latched_target)
            target[0] -= SYSTEM_CONFIG.PRE_GRASP_HEIGHT
            final_target_pos = np.array(target)
            
            if np.linalg.norm(current_ee_pos - final_target_pos) < 0.02:
                self.get_logger().info("⬇️ 到达预备点，开始下放")

        # ===========================
        # 🚀 平滑插值与 IK 执行
        # ===========================
        
        # 计算下一步的中间点 (限制单步位移，实现"缓缓移动")
        next_step_pos = self.smooth_move_to(
            current=self.last_target_pos, 
            target=final_target_pos, 
            step_size=SYSTEM_CONFIG.MAX_STEP_SIZE
        )
        self.last_target_pos = next_step_pos

        try:
            ik_kwargs = {
                'target_position': next_step_pos,
                'initial_position': full_joints,
                'target_orientation': target_orientation,
                'orientation_mode': orientation_mode 
            }

            ik_res = self.ik_chain.inverse_kinematics(**ik_kwargs)
            
            # 提取左臂关节数据
            target_cmds = [ik_res[i] for i, active in enumerate(self.active_mask) if active]
            final_cmds = list(target_cmds)
            final_cmds.append(gripper_cmd) 
            
            # 发布指令
            msg = Float64MultiArray()
            msg.data = final_cmds
            self.cmd_pub.publish(msg)

        except Exception as e:
            self.get_logger().error(f"IK Calculation Error: {e}")

def main():
    rclpy.init()
    node = DebugGraspController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
