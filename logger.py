# logger.py
import os
import threading
import csv
from datetime import datetime

class QueryLogger:
    def __init__(self, filename='query_log.csv'):
        self.filename = filename
        self.lock = threading.Lock()
        if not os.path.exists(filename):
            with open(filename, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['timestamp', 'query', 'predicted_tier', 'actual_tier', 'answer', 'latency', 'error'])

    def log(self, query: str, predicted_tier: str, actual_tier: str,
            answer: str, latency: float, error: str = ""):
        with self.lock:
            with open(self.filename, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([datetime.now().isoformat(), query, predicted_tier,
                                 actual_tier, answer, f"{latency:.4f}", error])