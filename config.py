# config.py
CONFIG = {
    # Core retrieval parameters
    "chunk_size": 500,
    "chunk_overlap": 100,
    "embedding_model": "paraphrase-MiniLM-L6-v2",
    "fusion_weights": {"semantic": 0.6, "keyword": 0.3, "metadata": 0.1},

    # Router settings
    "router_mode": "classifier_advanced",   # change to "rule_based", "classifier", or "classifier_advanced"
    "router_training_data": "router_training.csv",
    "router_model_path": "router_classifier_advanced.pkl",
    "router_confidence_threshold": 0.6,
    "fallback_threshold": 0.3,

    # ROS2 / Navigation
    "safety_costmap_service": "/global_costmap/get_costmap",
    "navigate_action": "/navigate_to_pose",
    "max_retries": 3,
    "goal_timeout_sec": 60.0,
    "costmap_threshold": 50,

    # Logging
    "log_file": "query_log.csv",

    # Ablation (optional, used by wrapper scripts)
    "ablation": {
        "chunk_sizes": [200, 500, 800, 1000],
        "chunk_overlaps": [50, 100, 150, 200],
        "embedding_models": ["paraphrase-MiniLM-L6-v2", "all-MiniLM-L12-v2", "all-mpnet-base-v2"],
        "fusion_weights": [
            {"semantic": 0.6, "keyword": 0.3, "metadata": 0.1},
            {"semantic": 0.7, "keyword": 0.2, "metadata": 0.1},
            {"semantic": 0.5, "keyword": 0.4, "metadata": 0.1},
            {"semantic": 0.4, "keyword": 0.3, "metadata": 0.3}
        ],
        "confidence_thresholds": [0.5, 0.6, 0.7, 0.8, 0.9]
    },

    "simulation": {
        "world": "warehouse.world",
        "robot_model": "turtlebot3_waffle",
        "num_robots": 1,
        "use_sim_time": True,
        "sensor_noise": {"enabled": True, "laser_stddev": 0.01, "odometry_stddev": 0.001},
        "dynamic_obstacles": {"enabled": False, "num_obstacles": 0, "speed_range": [0.1, 0.5]}
    }
}