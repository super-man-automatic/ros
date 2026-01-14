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
    OFFSET_Z =  0.00

    # 2. 安全避障参数
    SAFE_PARK_POS = [0.1, 0.2, 0.5] 
    SAFE_PARK_THRESHOLD = 0.05

    # 3. 抓取逻辑参数
    PRE_GRASP_DIS = 0.05   # 预备高度
    GRASP_HEIGHT_OFFSET = 0.02 # 抓取微调
    LIFT_HEIGHT = 0.20     # 抓取后抬起的高度
    
    GRIPPER_FORCE = 0.03    # 夹爪闭合力度 (仿真器将识别此值触发吸附)
    GRIPPER_OPEN = 0.8      # 夹爪张开力度

    # 4. 速度平滑参数
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
        
        self.ik_chain = None
        self.active_mask = []
        
        # 初始化模型
        self.init_ik_chain()
        
        self.current_joints = None
        self.last_target_pos = None 
        
        # 订阅与发布
        self.joint_sub = self.create_subscription(JointState, '/joint_states', self.joint_cb, 10)
        self.vision_sub = self.create_subscription(Float64MultiArray, '/openarm/target_object', self.vision_cb, 10)
        self.cmd_pub = self.create_publisher(Float64MultiArray, '/openarm/joint_commands', 10)
        
        self.state = 0
        self.pre_open_time = 0
        
        self.timer = self.create_timer(0.1, self.control_loop) 
        self.get_logger().info("🚀 抓取控制器启动！等待关节数据...")

    def init_ik_chain(self):
        xacro_file = "/home/zheng/openarm_ws/src/openarm_description/urdf/robot/v10.urdf.xacro"
        
        try:
            doc = xacro.process_file(xacro_file, mappings={'bimanual': 'true'})
            with tempfile.NamedTemporaryFile(suffix='.urdf', mode='w', delete=False) as tmp:
                tmp.write(doc.toprettyxml(indent='  '))
                urdf_path = tmp.name

            self.ik_chain = ikpy.chain.Chain.from_urdf_file(urdf_path, base_elements=["world"])
            
            self.active_mask = [False] * len(self.ik_chain.links)
            for i, link in enumerate(self.ik_chain.links):
                if "openarm_left_joint" in link.name: 
                    self.active_mask[i] = True
            self.ik_chain.active_links_mask = self.active_mask
            
            os.unlink(urdf_path)
            self.get_logger().info("✅ IK 链初始化成功")
            
        except Exception as e:
            self.get_logger().error(f"❌ 模型初始化失败: {e}")
            self.ik_chain = None

    def joint_cb(self, msg):
        try:
            left_arm_indices = [msg.name.index(f'openarm_left_joint{i}') for i in range(1, 8)]
            self.current_joints = [msg.position[i] for i in left_arm_indices]
        except ValueError: 
            pass

    def vision_cb(self, msg):
        if self.state != 1: return

        data = msg.data
        if len(data) < 3: return
        x, y, z = data[0], data[1], data[2]

        if abs(x) < 0.001 and abs(y) < 0.001: return

        final_pos = self.processor.process([x, y, z])

        self.latched_target = final_pos
        if len(self.processor.history) >= SYSTEM_CONFIG.FILTER_WINDOW_SIZE:
             self.get_logger().info(f"🎯 视觉锁定目标: {np.round(final_pos, 3)}")
             self.state = 2 

    def get_vertical_down_orientation(self):
        return np.array([
            [1,  0,  0],
            [0, -1,  0],
            [0,  0, -1] 
        ])
    
    def get_horizontal_forward_orientation(self):
        return np.array([
            [0,  0, 1],
            [0, -1, 0],
            [1,  0, 0]
        ])
    
    def smooth_move_to(self, current, target, step_size):
        diff = target - current
        dist = np.linalg.norm(diff)
        
        if dist <= step_size:
            return target
        else:
            direction = diff / dist
            return current + direction * step_size

    def control_loop(self):
        gripper_cmd = None
        if self.ik_chain is None:
            if self.state != -99:
                self.get_logger().error("⚠️ IK 未初始化，等待修复...")
                self.state = -99
            return

        if self.current_joints is None: return

        full_joints = [0.0] * len(self.ik_chain.links)
        idx = 0
        for i, active in enumerate(self.active_mask):
            if active and idx < len(self.current_joints):
                val = self.current_joints[idx]
                lower = self.ik_chain.links[i].bounds[0]
                upper = self.ik_chain.links[i].bounds[1]
                if val > upper: val = upper - 0.001
                full_joints[i] = val
                idx += 1
        
        current_pose_matrix = self.ik_chain.forward_kinematics(full_joints)
        current_ee_pos = current_pose_matrix[:3, 3]

        if self.last_target_pos is None:
            self.last_target_pos = current_ee_pos

        final_target_pos = current_ee_pos 
        
        target_orientation = self.get_horizontal_forward_orientation()
        orientation_mode = "all" 

        # ===========================
        # 🤖 状态机逻辑
        # ===========================
        
        if self.state == -1: # === 归位 ===
            gripper_cmd = SYSTEM_CONFIG.GRIPPER_OPEN
            msg = Float64MultiArray()
            cmd_joints = SYSTEM_CONFIG.HOME_JOINTS[:len(self.current_joints)]
            msg.data = cmd_joints + [SYSTEM_CONFIG.GRIPPER_OPEN]
            self.cmd_pub.publish(msg)
            
            curr = np.array(self.current_joints)
            tgt = np.array(cmd_joints)
            if np.linalg.norm(curr - tgt) < SYSTEM_CONFIG.HOME_THRESHOLD:
                self.get_logger().info("✅ 归位完成，准备前往安全高点...")
                self.processor.history.clear()
                self.last_target_pos = current_ee_pos 
                self.state = 0 
            return 

        elif self.state == 5: # === 抬起 ===
            # self.get_logger().info("state 5") # 保持原样注释
            # time.sleep(3) # 保持原样
            target = list(self.latched_target)
            target[2] += SYSTEM_CONFIG.LIFT_HEIGHT
            final_target_pos = np.array(target)
            gripper_cmd = SYSTEM_CONFIG.GRIPPER_FORCE
            
        elif self.state == 4: # === 抓取 ===
            # self.get_logger().info("state 4")
            target = list(self.latched_target)
            target[0] += SYSTEM_CONFIG.GRASP_HEIGHT_OFFSET
            final_target_pos = np.array(target)
            gripper_cmd = SYSTEM_CONFIG.GRIPPER_FORCE
            self.state = 5 
            
        elif self.state == 3: # === 下放 ===
            # self.get_logger().info("state 3")
            target = list(self.latched_target)
            target[0] += SYSTEM_CONFIG.GRASP_HEIGHT_OFFSET
            final_target_pos = np.array(target)
            gripper_cmd = SYSTEM_CONFIG.GRIPPER_OPEN 

            if np.linalg.norm(current_ee_pos - final_target_pos) < 0.01:
                self.get_logger().info("👌 到达抓取位")
                self.state = 4
                
        elif self.state == 2: # === 接近 ===
            target = list(self.latched_target)
            target[0] -= SYSTEM_CONFIG.PRE_GRASP_DIS
            final_target_pos = np.array(target)
            
            if np.linalg.norm(current_ee_pos - final_target_pos) < 0.02:
                self.get_logger().info("⬇️ 到达预备点，开始下放")
                self.state = 3
                
        elif self.state == 1: # === 视觉搜索 ===
            final_target_pos = np.array(SYSTEM_CONFIG.SAFE_PARK_POS)

        elif self.state == 0.5:  # === Pre-Open ===
            final_target_pos = np.array(SYSTEM_CONFIG.SAFE_PARK_POS)
            gripper_cmd = SYSTEM_CONFIG.GRIPPER_OPEN

            if time.time() - self.pre_open_time > 3:
                self.get_logger().info("⬇️ 夹爪已完全张开，开始下放")
                self.state = 1
                
        elif self.state == 0: # === 安全抬起 ===
            final_target_pos = np.array(SYSTEM_CONFIG.SAFE_PARK_POS)
            gripper_cmd = SYSTEM_CONFIG.GRIPPER_OPEN
            if np.linalg.norm(current_ee_pos - final_target_pos) < SYSTEM_CONFIG.SAFE_PARK_THRESHOLD:
                self.get_logger().info("🛡️ 已到达安全高点，开始视觉搜索...")
                self.state = 0.5 
                self.pre_open_time = time.time()

        # ===========================
        # 🚀 平滑插值与 IK 执行
        # ===========================
        next_step_pos = self.smooth_move_to(
            current=self.last_target_pos, 
            target=final_target_pos, 
            step_size=SYSTEM_CONFIG.MAX_STEP_SIZE
        )
        self.last_target_pos = next_step_pos
        
        if gripper_cmd is None:
            gripper_cmd = SYSTEM_CONFIG.GRIPPER_OPEN
            
        try:
            ik_kwargs = {
                'target_position': next_step_pos,
                'initial_position': full_joints,
                'target_orientation': target_orientation,
                'orientation_mode': orientation_mode 
            }

            ik_res = self.ik_chain.inverse_kinematics(**ik_kwargs)
            
            target_cmds = [ik_res[i] for i, active in enumerate(self.active_mask) if active]
            final_cmds = list(target_cmds)
            final_cmds.append(gripper_cmd) 
            
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
