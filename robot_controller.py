# robot_controller.py
import logging
import rclpy
from rclpy.action import ActionClient
from nav2_msgs.action import NavigateToPose
from geometry_msgs.msg import PoseStamped, Point, Quaternion

class RobotController:
    def __init__(self, node, action_name='/navigate_to_pose', timeout_sec=60.0, retries=3):
        self.node = node
        self.action_name = action_name
        self.timeout = timeout_sec
        self.retries = retries
        self.action_client = ActionClient(node, NavigateToPose, action_name)

    def send_goal(self, x: float, y: float, z: float = 0.0,
                  orientation=(0.0, 0.0, 0.0, 1.0),
                  frame_id: str = 'map') -> bool:
        for attempt in range(self.retries):
            goal = NavigateToPose.Goal()
            goal.pose.header.frame_id = frame_id
            goal.pose.header.stamp = self.node.get_clock().now().to_msg()
            goal.pose.pose.position = Point(x=x, y=y, z=z)
            goal.pose.pose.orientation = Quaternion(x=orientation[0], y=orientation[1],
                                                    z=orientation[2], w=orientation[3])

            if not self.action_client.wait_for_server(timeout_sec=5.0):
                logging.warning(f"Action server {self.action_name} not available. Retry {attempt+1}")
                continue

            future = self.action_client.send_goal_async(goal)
            rclpy.spin_until_future_complete(self.node, future)
            goal_handle = future.result()

            if not goal_handle.accepted:
                logging.warning(f"Goal rejected. Retry {attempt+1}")
                continue

            result_future = goal_handle.get_result_async()
            rclpy.spin_until_future_complete(self.node, result_future, timeout_sec=self.timeout)
            result = result_future.result()
            if result and result.code == NavigateToPose.Result.SUCCEEDED:
                logging.info(f"Goal reached after {attempt+1} attempts.")
                return True
            else:
                logging.warning(f"Goal failed. Retry {attempt+1}")

        logging.error("All retries exhausted.")
        return False