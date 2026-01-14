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

# [修改点 1] 请在此处填写你 XML 文件中那个想要抓取的物体 Body 的名称
# 如果不知道名字，在 Mujoco Viewer 右侧 Body 列表中查看
GRASP_OBJECT_NAME = "apple"  # <--- 请根据实际 XML 修改此处

# [修改点 2] 夹爪末端的 Link 名称 (通常是第7轴或 palm)
GRIPPER_LINK_NAME = "openarm_left_link7" 
# ===========================================

class MujocoRosBridge(Node):
    def __init__(self):
        super().__init__('mujoco_ros_bridge')
        
        # 1. 发布器
        self.image_pub = self.create_publisher(Image, '/camera/rgb/image_raw', 10)
        self.joint_pub = self.create_publisher(JointState, '/joint_states', 10)
        self.cam_ext_pub = self.create_publisher(Float64MultiArray, '/camera/extrinsics', 10)
        
        # 2. 订阅控制指令
        self.ctrl_sub = self.create_subscription(
            Float64MultiArray, 
            '/openarm/joint_commands', 
            self.ctrl_callback, 
            10)
        
        self.bridge = CvBridge()


        # 3. 加载模型
        try:
            self.model = mujoco.MjModel.from_xml_path(XML_PATH)
            self.data = mujoco.MjData(self.model)
        except Exception as e:

            return
        
        # 4. 初始化索引
        self._init_indices()

        # 5. 初始化抓取相关变量 [新增]
        self.is_grasped = False
        self.grasp_offset = np.zeros(3)
        self.object_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, GRASP_OBJECT_NAME)
        self.gripper_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, GRIPPER_LINK_NAME)
        
        if self.object_body_id == -1:
            self.get_logger().warn(f"⚠️ 未找到物体 '{GRASP_OBJECT_NAME}'，吸附功能将无效")

        # 初始目标
        current_init_pos = self.data.qpos[self.arm_qpos_indices].copy()
        self.target_arm_pos = current_init_pos
        self.target_gripper_val = 10.0 # 默认张开

        # 渲染器
        self.renderer = mujoco.Renderer(self.model, height=480, width=640)
        self.camera_name = "top_camera" 

        self.run_simulation()

    def _init_indices(self):
        self.arm_qpos_indices = []
        self.arm_qvel_indices = []
        self.gripper_ctrl_indices = []
        
        for name in ARM_JOINT_NAMES:
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if jid != -1: 
                self.arm_qpos_indices.append(self.model.jnt_qposadr[jid])
                self.arm_qvel_indices.append(self.model.jnt_dofadr[jid])

        for name in GRIPPER_ACTUATOR_NAMES:
            act_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
            if act_id != -1: self.gripper_ctrl_indices.append(act_id)

    def ctrl_callback(self, msg):
        data = msg.data
        if len(data) >= len(self.arm_qpos_indices):
            self.target_arm_pos = np.array(data[0:len(self.arm_qpos_indices)])
        if len(data) > len(self.arm_qpos_indices):
            self.target_gripper_val = data[len(self.arm_qpos_indices)]

    def run_simulation(self):
        dt = self.model.opt.timestep
        smoothed_target = self.data.qpos[self.arm_qpos_indices].copy()
        alpha = 0.05 

        with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
            while viewer.is_running() and rclpy.ok():
                step_start = time.time()

                # --- 原有逻辑：平滑插值与机械臂控制 ---
                raw_target = self.target_arm_pos
                smoothed_target = (1 - alpha) * smoothed_target + alpha * raw_target

                current_qpos = self.data.qpos[self.arm_qpos_indices].copy()
                calculated_vel = (smoothed_target - current_qpos) / dt
                
                self.data.qpos[self.arm_qpos_indices] = smoothed_target
                self.data.qvel[self.arm_qvel_indices] = calculated_vel
                
                for act_id in self.gripper_ctrl_indices:
                    self.data.ctrl[act_id] = self.target_gripper_val

                # --- [新增] 吸附 (Magic Grasp) 逻辑 ---
                if self.object_body_id != -1 and self.gripper_body_id != -1:
                    # 判断指令：你的代码中，0.027为闭合，10.0为张开
                    # 所以 < 1.0 认为是闭合指令
                    is_close_cmd = (self.target_gripper_val < 0.5)
                    
                    ee_pos = self.data.xpos[self.gripper_body_id] # 末端位置
                    obj_pos = self.data.xpos[self.object_body_id] # 物体位置
                    
                    # 1. 触发抓取 (状态翻转)
                    if is_close_cmd and not self.is_grasped:
                        dist = np.linalg.norm(ee_pos - obj_pos)
                        if dist < 0.3: # 距离小于 15cm 允许吸附
                            self.is_grasped = True
                            self.grasp_offset = obj_pos - ee_pos # 记录相对偏移


                    # 2. 触发松开
                    elif not is_close_cmd and self.is_grasped:
                        self.is_grasped = False


                    # 3. 持续吸附 (修改物体物理状态)
                    if self.is_grasped:
                        # 找到物体对应的 Joint 数据地址
                        # 注意：这里假设物体是 FreeJoint，我们修改它的 qpos (前3位是 XYZ)
                        obj_jnt_adr = self.model.body_jntadr[self.object_body_id]
                        if obj_jnt_adr != -1:
                            qpos_adr = self.model.jnt_qposadr[obj_jnt_adr]
                            qvel_adr = self.model.jnt_dofadr[obj_jnt_adr]
                            
                            # 强制修改位置：末端 + 偏移
                            target_pos = ee_pos + self.grasp_offset
                            self.data.qpos[qpos_adr:qpos_adr+3] = target_pos
                            
                            # 强制修改速度为0 (防止飞出)
                            self.data.qvel[qvel_adr:qvel_adr+6] = 0.0

                # --- 物理步进 ---
                mujoco.mj_step(self.model, self.data)
                
                # --- 原有逻辑：发布与同步 ---
                if self.data.time % 0.033 < dt: 
                    self.publish_joint_state()
                    self.publish_camera_data() 

                viewer.sync()
                
                time_diff = time.time() - step_start
                if time_diff < dt:
                    time.sleep(dt - time_diff)
                
                rclpy.spin_once(self, timeout_sec=0.0)

    def publish_joint_state(self):
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = ARM_JOINT_NAMES 
        positions = [self.data.qpos[i] for i in self.arm_qpos_indices]
        msg.position = positions
        self.joint_pub.publish(msg)

    def publish_camera_data(self):
        self.renderer.update_scene(self.data, camera=self.camera_name)
        rgb = self.renderer.render()
        try:
            img_msg = self.bridge.cv2_to_imgmsg(rgb, encoding="rgb8")
            img_msg.header.stamp = self.get_clock().now().to_msg()
            self.image_pub.publish(img_msg)
        except:
            pass

        cam_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, self.camera_name)
        if cam_id != -1:
            pos = self.data.cam_xpos[cam_id] 
            mat = self.data.cam_xmat[cam_id] 
            ext_msg = Float64MultiArray()
            ext_msg.data = np.concatenate([pos, mat]).tolist()
            self.cam_ext_pub.publish(ext_msg)

def main(args=None):
    rclpy.init(args=args)
    MujocoRosBridge()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
