#!/usr/bin/env python3

import rclpy

from rclpy.node import Node

from sensor_msgs.msg import Image

from geometry_msgs.msg import Point

from cv_bridge import CvBridge

import cv2

import numpy as np



class VisionTracker(Node):

    def __init__(self):

        super().__init__('vision_tracker')

        

        # 订阅与发布

        self.subscription = self.create_subscription(

            Image, '/camera/rgb/image_raw', self.image_callback, 10)

        self.coord_pub = self.create_publisher(Point, '/openarm/apple_position', 10)

        

        self.bridge = CvBridge()

        self.get_logger().info("视觉节点 V3 (流畅版) 已启动...")



        # --- 坐标校准参数 ---

        # 既然你看到的是 0.8，而真实是 0.6，我们需要减去 0.2

        self.offset_x = -0.20  

        self.offset_y = 0.0    # Y为-0.18，非常接近-0.2，暂时不修

        

        # 比例系数

        self.scale_x = 0.0012

        self.scale_y = 0.0012



        # --- 优化：只在初始化时创建一次窗口，解决卡顿问题 ---

        cv2.namedWindow("Vision Tracker", cv2.WINDOW_NORMAL)

        cv2.resizeWindow("Vision Tracker", 800, 600)



    def image_callback(self, msg):

        try:

            # 1. 图像转换

            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")

            

            # 2. 旋转 (保持横向)

            cv_image = cv2.rotate(cv_image, cv2.ROTATE_90_CLOCKWISE)

            

            # 获取参数

            H, W = cv_image.shape[:2]

            cx, cy = W // 2, H // 2



            # 3. 颜色识别

            hsv = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)

            mask1 = cv2.inRange(hsv, np.array([0, 100, 100]), np.array([10, 255, 255]))

            mask2 = cv2.inRange(hsv, np.array([170, 100, 100]), np.array([180, 255, 255]))

            mask = mask1 + mask2

            

            # 4. 轮廓查找

            contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            

            # 默认坐标 (如果没有识别到)

            calc_x, calc_y, calc_z = 0.0, 0.0, 0.0



            if contours:

                c = max(contours, key=cv2.contourArea)

                ((u, v), radius) = cv2.minEnclosingCircle(c)

                

                if radius > 5:

                    # 画图

                    cv2.circle(cv_image, (int(u), int(v)), int(radius), (0, 255, 255), 2)

                    cv2.circle(cv_image, (int(u), int(v)), 5, (0, 0, 255), -1)

                    

                    # 5. 坐标计算

                    # X: 垂直方向 (v)

                    raw_x = 0.6 + (v - cy) * self.scale_x

                    # Y: 水平方向 (u)

                    raw_y = 0.0 - (u - cx) * self.scale_y



                    # 应用校准

                    calc_x = raw_x + self.offset_x

                    calc_y = raw_y + self.offset_y

                    calc_z = 0.45 # 固定高度



                    # 发布坐标

                    point = Point()

                    point.x, point.y, point.z = calc_x, calc_y, calc_z

                    self.coord_pub.publish(point)

                    

                    # 6. 显示文字 (包含 XYZ)

                    text_str = f"X:{calc_x:.2f} Y:{calc_y:.2f} Z:{calc_z:.2f}"

                    cv2.putText(cv_image, text_str, (0, int(v)-20), 

                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)



            # 7. 刷新图像 (不再重复创建窗口)

            cv2.imshow("Vision Tracker", cv_image)

            cv2.waitKey(1) # 必须有这一句，否则窗口无响应

            

        except Exception as e:

            self.get_logger().error(f"Err: {e}")



def main(args=None):

    rclpy.init(args=args)

    node = VisionTracker()

    try:

        rclpy.spin(node)

    except KeyboardInterrupt:

        pass

    node.destroy_node()

    rclpy.shutdown()

    cv2.destroyAllWindows()



if __name__ == '__main__':

    main()
