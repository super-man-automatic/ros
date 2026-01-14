#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray
import time
cmd = [0.0] * 7
class CalibrationTest(Node):
    def __init__(self):
        super().__init__('calibration_test')
        self.pub = self.create_publisher(Float64MultiArray, '/openarm/joint_commands', 10)
        self.timer = self.create_timer(1.0, self.loop)
        self.step = 0
        self.get_logger().info("标定测试开始！请观察 MuJoCo 中的机器人动作...")

    def loop(self):
        print('current msg:')
        print(cmd)
        print('change which:')
        id=input()
        try:
            id=int(id)
            cmd[id]=1.0*float(input('change to:'))
        except Exception as e:
            print(e)
        # 发送指令 (双臂，右臂在前)
        msg = Float64MultiArray()
        msg.data = cmd + [0.0]*10
        self.pub.publish(msg)
        

	
def main(args=None):
    rclpy.init(args=args)
    node = CalibrationTest()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
