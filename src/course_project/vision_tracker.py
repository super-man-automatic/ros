#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Float64MultiArray
from cv_bridge import CvBridge
import cv2
import numpy as np

class AppleDetectorSubscriber(Node):
    def __init__(self):
        super().__init__('apple_detector_sub')
        
        self.width = 640
        self.height = 480
        self.bridge = CvBridge()

        # [参数] 相机内参 (假设FOV不变，如果变了也可以从Bridge发过来)
        self.fovy = 58.0
        self.f = (self.height / 2.0) / np.tan(np.deg2rad(self.fovy) / 2.0)
        
        # [状态] 相机外参 (初始化为空，等待 Bridge 发送)
        self.cam_pos = None
        self.cam_mat = None

        self.FIXED_Z = 0.48

        # 1. 订阅图像
        self.sub_img = self.create_subscription(
            Image, '/camera/rgb/image_raw', self.image_callback, 10)
        
        # 2. [新增] 订阅相机外参 (自动标定)
        self.sub_ext = self.create_subscription(
            Float64MultiArray, '/camera/extrinsics', self.extrinsics_callback, 10)

        # 3. [新增] 发布目标坐标 (发给控制节点去抓取)
        # 格式: [x, y, z]
        self.target_pub = self.create_publisher(Float64MultiArray, '/openarm/target_object', 10)
        
        self.get_logger().info("视觉节点已启动，等待相机参数...")

    def extrinsics_callback(self, msg):
        """接收来自仿真的真实相机位姿"""
        data = np.array(msg.data)
        if len(data) == 12:
            self.cam_pos = data[0:3]       # 前3位是位置
            self.cam_mat = data[3:].reshape(3, 3) # 后9位是矩阵

    def map_ccw_rotated_to_original(self, u_rot, v_rot):
        """处理图像旋转后的坐标映射"""
        # 旋转前(Original): 宽640, 高480
        # 旋转后(Rotated):  宽480, 高640 (u_rot, v_rot)
        # 逆时针90度关系:
        v_orig = u_rot
        u_orig = (self.width - 1) - v_rot
        return u_orig, v_orig

    def pixel_to_world_fixed_z(self, u, v):
        if self.cam_pos is None or self.cam_mat is None:
            return None

        # 1. 构造 OpenCV 标准相机坐标系下的向量 (Z=1, Looking Forward)
        # 注意：这里直接用 u, v (原始未旋转图的坐标)
        cx = self.width / 2.0
        cy = self.height / 2.0
        
        # 归一化平面坐标 (z=1)
        # OpenCV 定义: x向右, y向下, z向前
        z_c = 1.0
        x_c = (u - cx) / self.f * z_c
        y_c = (v - cy) / self.f * z_c
        
        vec_opencv = np.array([x_c, y_c, z_c])

        # 2. 定义变换矩阵：OpenCV (Z-fwd, Y-down) -> MuJoCo (Z-back, Y-up)
        # X轴不变(1)，Y轴翻转(-1)，Z轴翻转(-1)
        R_opencv_to_mujoco = np.array([
            [1,  0,  0],
            [0, -1,  0],
            [0,  0, -1]
        ])

        # 3. 转换到 MuJoCo 相机本体坐标系
        vec_mujoco_local = R_opencv_to_mujoco @ vec_opencv

        # 4. 转换到世界坐标系 (应用外参旋转矩阵)
        # self.cam_mat 是 MuJoCo 的 cam_xmat (Local -> World)
        vec_world_dir = self.cam_mat @ vec_mujoco_local
        
        # 归一化方向向量
        vec_world_dir = vec_world_dir / np.linalg.norm(vec_world_dir)

        # 5. 射线求交 (计算 t)
        # P_world = P_cam + t * V_dir
        # target_z = cam_z + t * dir_z  =>  t = (target_z - cam_z) / dir_z
        
        if abs(vec_world_dir[2]) < 1e-6:
            return None # 平行，无交点

        t = (self.FIXED_Z - self.cam_pos[2]) / vec_world_dir[2]

        if t < 0:
            # 物体在相机背面 (通常不会发生，除非标定极度错误)
            return None

        final_x = self.cam_pos[0] + t * vec_world_dir[0]
        final_y = self.cam_pos[1] + t * vec_world_dir[1]
        
        return np.array([final_x, final_y, self.FIXED_Z])
        
    def image_callback(self, msg):
        if self.cam_pos is None:
            # 如果还没收到外参，先不处理
            return

        try:
            cv_img = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            # 旋转 (注意：旋转是为了显示方便，还是安装方式导致？假设是安装导致图像侧着)
            cv_img_rotated = cv2.rotate(cv_img, cv2.ROTATE_90_COUNTERCLOCKWISE)
            
            hsv = cv2.cvtColor(cv_img_rotated, cv2.COLOR_BGR2HSV)
            
            # 红色阈值
            lower1, upper1 = np.array([0, 40, 40]), np.array([10, 255, 255])
            lower2, upper2 = np.array([170, 40, 40]), np.array([180, 255, 255])
            mask = cv2.inRange(hsv, lower1, upper1) + cv2.inRange(hsv, lower2, upper2)
            
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            target_pos = None
            if contours:
                c = max(contours, key=cv2.contourArea)
                if cv2.contourArea(c) > 20:
                    M = cv2.moments(c)
                    if M["m00"] != 0:
                        cX_rot = int(M["m10"] / M["m00"])
                        cY_rot = int(M["m01"] / M["m00"])
                        
                        # 1. 还原到原始传感器像素坐标 (640x480)
                        u_orig, v_orig = self.map_ccw_rotated_to_original(cX_rot, cY_rot)
                        
                        # 2. 结合 Bridge 发来的外参进行 3D 解算
                        target_pos = self.pixel_to_world_fixed_z(u_orig, v_orig)
                        
                        # 画图
                        cv2.circle(cv_img_rotated, (cX_rot, cY_rot), 10, (0, 255, 0), 2)

            # 显示和发布结果
            if target_pos is not None:

                text = f"XY: {target_pos[0]:.3f}, {target_pos[1]:.3f}"
                cv2.putText(cv_img_rotated, text, (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # 发布给机械臂控制节点
                out_msg = Float64MultiArray()
                out_msg.data = target_pos.tolist()
                out_msg.data[2]=self.FIXED_Z
                self.target_pub.publish(out_msg)

            cv2.imshow("Vision (Auto Calibrated)", cv_img_rotated)
            cv2.waitKey(1)

        except Exception as e:
            self.get_logger().error(f"Error: {e}")

def main(args=None):
    rclpy.init(args=args)
    node = AppleDetectorSubscriber()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
