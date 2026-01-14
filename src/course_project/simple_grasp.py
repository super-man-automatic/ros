#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray
from geometry_msgs.msg import Point
import time
import math

class DebugGrasp(Node):
    def __init__(self):
        super().__init__('debug_grasp')
        self.pub = self.create_publisher(Float64MultiArray, '/openarm/joint_commands', 10)
        self.timer = self.create_timer(0.05, self.control_loop)
        self.start_time = time.time()
        
        # === !!! 调试目标 (强制设定，忽略视觉) !!! ===
        self.debug_x = 0.5   # 正前方 0.5米
        self.debug_y = 0.00   # 正中间
        self.debug_z = 0.45   # 苹果高度

        # === !!! 机械臂参数 (在这里微调) !!! ===
        self.L_SHOULDER = 0.28
        self.L_FOREARM  = 0.22
        self.L_HAND     = 0.18
        self.ROBOT_BASE_Z = 0.76

        # === !!! 关键修正：角度方向 !!! ===
        # 如果机器人向侧面转，修改这个 offset (比如加减 1.57 = 90度)
        self.BASE_OFFSET =0
        
        # 如果大臂是向后仰而不是向前伸，把这个符号改成 -1.0
        self.SHOULDER_DIR = -1.0 
        
        self.get_logger().info("调试模式启动！目标: 正前方 (0.5, 0.0)")

    def calculate_ik(self, x, y, z):
        # 1. 底座: 增加 Offset 修正“向侧面伸”的问题
        theta_base = math.atan2(y, x) + self.BASE_OFFSET
        
        planar_dist = math.sqrt(x**2 + y**2)
        wrist_r = planar_dist - self.L_HAND
        wrist_z = z - self.ROBOT_BASE_Z
        D = math.sqrt(wrist_r**2 + wrist_z**2)
        
        if D > (self.L_SHOULDER + self.L_FOREARM):
            print("太远了！")
            return None

        cos_elbow = (self.L_SHOULDER**2 + self.L_FOREARM**2 - D**2) / (2 * self.L_SHOULDER * self.L_FOREARM)                                                          
        cos_elbow = max(-1.0, min(1.0, cos_elbow))
        
        # 2. 肘部: 修正“幅度不够”的问题
        # 试着把这里的 pi 减去 改成 直接用 angle，或者反过来                      
        angle_elbow_inner = math.acos(cos_elbow)
        theta_elbow = angle_elbow_inner # 尝试方案A: 直接用内角
        # theta_elbow = math.pi - angle_elbow_inner # 方案B: 互补角 (如果方案A导致手臂反向弯曲，就换这个)
        
        # 3. 肩部: 修正方向
        angle_elevation = math.atan2(wrist_z, wrist_r)
        cos_shoulder = (self.L_SHOULDER**2 + D**2 - self.L_FOREARM**2) / (2 * self.L_SHOULDER * D)
        cos_shoulder = max(-1.0, min(1.0, cos_shoulder))
        angle_shoulder_inner = math.acos(cos_shoulder)
        
        # 乘以方向系数
        theta_shoulder = -1.0 * self.SHOULDER_DIR * (angle_elevation + angle_shoulder_inner)
        
        # 4. 手腕: 自动放平
        # 这里的公式取决于上面的正负号，可能需要微调
        theta_wrist = -(theta_shoulder + theta_elbow)

        return [theta_base, theta_shoulder, theta_elbow, theta_wrist]

    def control_loop(self):
        t = time.time() - self.start_time
        cmd = [0.0] * 7
        
        # 简单的测试动作
        if t < 12.0:
            print("归位...", end='\r')
            # 这是一个关键的测试：Base=0, Shoulder=0... 看看机器人“零位”长什么样
            # 正常来说应该是指向正前方的直立状态，或者是平躺
            cmd = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        else:
            print(f"尝试去抓: ({self.debug_x}, {self.debug_y})", end='\r')
            angles = self.calculate_ik(self.debug_x, self.debug_y, self.debug_z)
            if angles:
                cmd[0], cmd[1], cmd[3], cmd[5] = angles
                
                # 强制修正肘部: 如果发现幅度不够，可能是肘部没伸开
                # 这里手动加一个补偿看看
                cmd[3] += 0.5 

        # 发送
        msg = Float64MultiArray()
        msg.data = cmd + [0.0]*10
        self.pub.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    Node = DebugGrasp()
    rclpy.spin(Node)
    Node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
