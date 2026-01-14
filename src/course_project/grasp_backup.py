#!/usr/bin/env python3    
import rclpy
from rclpy.node import Node
import numpy as np
import ikpy.chain
import xacro
import tempfile
import os
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray
from ament_index_python.packages import get_package_share_directory

class DebugGraspController(Node):
    def __init__(self):
        super().__init__('debug_grasp_controller')
        
        # 1. 初始化模型与IK链
        self.init_ik_chain()
        
        # 2. 订阅当前关节状态 (用于IK初值，防止跳变)
        self.current_joints = None
        self.joint_sub = self.create_subscription(JointState, '/joint_states', self.joint_cb, 10)
        
        # 3. 发布器 (根据你的接口调整，这里假设是直接发布到 position_controllers)
        self.cmd_pub = self.create_publisher(Float64MultiArray, '/openarm/joint_commands', 10)
        
        # 4. 调试参数
        self.test_stage = 0  # 0:停顿, 1:沿X移动, 2:沿Y, 3:沿Z
        self.timer = self.create_timer(2.0, self.debug_loop) # 每2秒变换一个动作
        self.get_logger().info("🚀 调试节点已启动，准备进行坐标轴偏移测试...")

    def init_ik_chain(self):
        # 定位入口 Xacro
        xacro_file = "/home/zheng/openarm_ws/src/openarm_description/urdf/robot/v10.urdf.xacro"
        try:
            # 转换 Xacro 为 URDF，显式开启双臂
            doc = xacro.process_file(xacro_file, mappings={'bimanual': 'true'})
            with tempfile.NamedTemporaryFile(suffix='.urdf', mode='w', delete=False) as tmp:
                tmp.write(doc.toprettyxml())
                urdf_path = tmp.name

            # 加载 IK 链，注意 base_elements 必须对应 Xacro 里的 body_connected_to
            self.ik_chain = ikpy.chain.Chain.from_urdf_file(urdf_path, base_elements=["world"])
            
            # 配置 Mask：只激活左臂的 7 个关节
            self.active_mask = [False] * len(self.ik_chain.links)
            for i, link in enumerate(self.ik_chain.links):
                if "openarm_left_joint" in link.name:
                    self.active_mask[i] = True
            
            self.ik_chain.active_links_mask = self.active_mask
            self.get_logger().info(f"✅ IK链加载成功，激活关节数: {sum(self.active_mask)}")
            os.unlink(urdf_path)
        except Exception as e:
            self.get_logger().error(f"❌ 模型初始化失败: {e}")

    def joint_cb(self, msg):
        # 提取左臂的关节位置
        # 注意：需要根据你的实际 msg.name 顺序进行映射
        left_arm_indices = [msg.name.index(f'openarm_left_joint{i}') for i in range(1, 8)]
        self.current_joints = [msg.position[i] for i in left_arm_indices]

    def debug_loop(self):
        if self.current_joints is None:
            self.get_logger().warn("等待关节状态数据...")
            return

        # 获取当前末端位姿 (Forward Kinematics)
        # ikpy 需要完整的关节列表长度
        full_joints = [0.0] * len(self.ik_chain.links)
        # 将当前关节填充进全链路
        idx = 0
        for i, active in enumerate(self.active_mask):
            if active:
                full_joints[i] = self.current_joints[idx]
                idx += 1
        
        current_pose_matrix = self.ik_chain.forward_kinematics(full_joints)
        current_pos = current_pose_matrix[:3, 3]
        
        # 设定目标点：在当前位置基础上做微调
        target_pos = current_pos.copy()
        
        if self.test_stage == 0:
            self.get_logger().info(f"📍 当前末端位置 (米): {current_pos}")
            self.test_stage = 1
            return
        elif self.test_stage == 1:
            self.get_logger().info("🧪 测试 1: 尝试沿 X 轴正向移动 10cm")
            target_pos[0] += 0.1 
        elif self.test_stage == 2:
            self.get_logger().info("🧪 测试 2: 尝试沿 Y 轴正向移动 10cm")
            target_pos[1] += 0.1
        elif self.test_stage == 3:
            self.get_logger().info("🧪 测试 3: 尝试沿 Z 轴正向移动 10cm")
            target_pos[2] += 0.1
            self.test_stage = -1 # 循环结束
        
        self.test_stage += 1

        # 执行逆运动学求解
        target_matrix = np.eye(4)
        target_matrix[:3, 3] = target_pos
        target_matrix[:3, :3] = current_pose_matrix[:3, :3] # 保持当前姿态不变

        # 核心修复：传入 initial_position 防止关节角度跳变导致的“摇摆”
        ik_res = self.ik_chain.inverse_kinematics(
            target_matrix, 
            initial_position=full_joints
        )

        # 提取 7 个目标关节角
        target_cmds = [ik_res[i] for i, active in enumerate(self.active_mask) if active]
        
        # 发布指令
        msg = Float64MultiArray()
        msg.data = target_cmds
        self.cmd_pub.publish(msg)

def main():
    rclpy.init()
    node = DebugGraspController()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
