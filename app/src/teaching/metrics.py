from typing import Dict, List
from datetime import datetime

class EngagementMetrics:
    def __init__(self):
        self.metrics = {
            "questions_asked": 0,
            "responses_given": 0,
            "comprehension_checks": [],
            "topic_durations": {},
            "session_start": datetime.now()
        }
    
    def record_question(self):
        self.metrics["questions_asked"] += 1
    
    def record_response(self):
        self.metrics["responses_given"] += 1
    
    def record_comprehension_check(self, result: bool):
        self.metrics["comprehension_checks"].append({
            "timestamp": datetime.now(),
            "correct": result
        })
    
    def get_metrics(self) -> Dict:
        return self.metrics.copy()