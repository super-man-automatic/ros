#!/usr/bin/env python3
import time
import mujoco
import mujoco.viewer
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, JointState
from std_msgs.msg import Float64MultiArray
from cv_bridge import CvBridge

# ================= 配置区域 =================
XML_PATH = "/home/zheng/openarm_ws/src/openarm_mujoco/v1/course_scene.xml"
ARM_JOINT_NAMES = [
    "openarm_left_joint1", "openarm_left_joint2", "openarm_left_joint3",
    "openarm_left_joint4", "openarm_left_joint5", "openarm_left_joint6",
    "openarm_left_joint7"
]
GRIPPER_ACTUATOR_NAMES = ["left_finger1_ctrl", "left_finger2_ctrl"]
# ===========================================

class MujocoRosBridge(Node):
    def __init__(self):
        super().__init__('mujoco_ros_bridge')
        
        # 1. 发布器
        self.image_pub = self.create_publisher(Image, '/camera/rgb/image_raw', 10)
        self.joint_pub = self.create_publisher(JointState, '/joint_states', 10)
        
        # [新增] 发布相机外参 (位置3 + 旋转9 = 12个float)
        self.cam_ext_pub = self.create_publisher(Float64MultiArray, '/camera/extrinsics', 10)
        
        # 2. 订阅控制指令
        self.ctrl_sub = self.create_subscription(
            Float64MultiArray, 
            '/openarm/joint_commands', 
            self.ctrl_callback, 
            10)
        
        self.bridge = CvBridge()
        self.get_logger().info("MuJoCo-ROS2 Bridge (带自动标定参数) 已启动！")

        # 3. 加载模型
        try:
            self.model = mujoco.MjModel.from_xml_path(XML_PATH)
            self.data = mujoco.MjData(self.model)
        except Exception as e:
            self.get_logger().error(f"加载 XML 失败: {e}")
            return
        
        # 4. 初始化索引
        self._init_indices()

        # 初始化控制目标
        self.target_arm_pos = np.zeros(len(ARM_JOINT_NAMES))
        self.target_gripper_force = 0.0

        # 渲染器
        self.renderer = mujoco.Renderer(self.model, height=480, width=640)
        self.camera_name = "top_camera" # 确保XML里有这个相机

        self.run_simulation()

    def _init_indices(self):
        self.arm_qpos_indices = []
        self.gripper_ctrl_indices = []
        for name in ARM_JOINT_NAMES:
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if jid != -1: self.arm_qpos_indices.append(self.model.jnt_qposadr[jid])
        for name in GRIPPER_ACTUATOR_NAMES:
            act_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
            if act_id != -1: self.gripper_ctrl_indices.append(act_id)

    def ctrl_callback(self, msg):
        data = msg.data
        if len(data) >= len(self.arm_qpos_indices):
            self.target_arm_pos = data[0:len(self.arm_qpos_indices)]
        if len(data) > len(self.arm_qpos_indices):
            self.target_gripper_force = data[len(self.arm_qpos_indices)]

    def run_simulation(self):
        with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
            while viewer.is_running() and rclpy.ok():
                step_start = time.time()

                # --- 物理步进 ---
                for i, q_adr in enumerate(self.arm_qpos_indices):
                    self.data.qpos[q_adr] = self.target_arm_pos[i]
                for act_id in self.gripper_ctrl_indices:
                    self.data.ctrl[act_id] = self.target_gripper_force
                mujoco.mj_step(self.model, self.data)
                
                # --- 降频发布 (30Hz) ---
                if self.data.time % 0.033 < 0.002: 
                    self.publish_joint_state()
                    self.publish_camera_data() # 合并了图像和参数发布

                viewer.sync()
                
                time_diff = time.time() - step_start
                if time_diff < self.model.opt.timestep:
                    time.sleep(self.model.opt.timestep - time_diff)
                
                rclpy.spin_once(self, timeout_sec=0.0)

    def publish_joint_state(self):
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = ARM_JOINT_NAMES 
        positions = [self.data.qpos[i] for i in self.arm_qpos_indices]
        msg.position = positions
        self.joint_pub.publish(msg)

    def publish_camera_data(self):
        # 1. 更新并发布图像
        self.renderer.update_scene(self.data, camera=self.camera_name)
        rgb = self.renderer.render()
        try:
            img_msg = self.bridge.cv2_to_imgmsg(rgb, encoding="rgb8")
            img_msg.header.stamp = self.get_clock().now().to_msg()
            self.image_pub.publish(img_msg)
        except:
            pass

        # 2. [关键修改] 获取并发布相机真实外参
        # 这样 Vision 节点就能拿到完美的标定数据
        cam_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, self.camera_name)
        if cam_id != -1:
            # MuJoCo 的相机位置 (x,y,z)
            pos = self.data.cam_xpos[cam_id] 
            # MuJoCo 的相机旋转矩阵 (3x3 平铺为 9)
            mat = self.data.cam_xmat[cam_id] 
            
            # 打包发送: 前3位是pos, 后9位是mat
            ext_msg = Float64MultiArray()
            ext_msg.data = np.concatenate([pos, mat]).tolist()
            self.cam_ext_pub.publish(ext_msg)

def main(args=None):
    rclpy.init(args=args)
    MujocoRosBridge()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
