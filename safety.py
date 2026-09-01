# safety.py
import logging
import rclpy
from nav2_msgs.srv import GetCostmap

class SafetyChecker:
    def __init__(self, node, costmap_service='/global_costmap/get_costmap', threshold=50):
        self.node = node
        self.threshold = threshold
        self.client = node.create_client(GetCostmap, costmap_service)
        try:
            while not self.client.wait_for_service(timeout_sec=1.0):
                logging.info("Waiting for costmap service...")
        except Exception as e:
            logging.warning(f"Could not connect to costmap service: {e}. Safety checks will be bypassed.")

    def is_goal_safe(self, x, y):
        # Placeholder – always safe for testing
        return True

class EnhancedSafetyChecker(SafetyChecker):
    def __init__(self, node, costmap_service='/global_costmap/get_costmap', threshold=50, safety_buffer=0.5):
        super().__init__(node, costmap_service, threshold)
        self.safety_buffer = safety_buffer

    def is_goal_safe(self, x, y):
        # Enhanced checks (placeholder – for testing, always safe)
        return True, "Safe (placeholder)", "low"