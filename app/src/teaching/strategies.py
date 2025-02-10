from typing import Dict, List
import re

class TeachingStrategy:
    @staticmethod
    def generate_comprehension_check(topic: str, content: str) -> str:
        """Generate a comprehension check question"""
        return f"""
        Based on what we've discussed about {topic}, please answer this quick question:
        What is the main concept we just covered in {content}?
        """
    
    @staticmethod
    def create_topic_introduction(topic: str, content: str) -> str:
        """Create an engaging introduction for a new topic"""
        return f"""
        I'm excited to help you learn about {topic}! 
        Let's start by understanding the basic concepts:
        {content}
        """
    
    @staticmethod
    def analyze_student_response(response: str) -> Dict:
        """Analyze student response for engagement and understanding"""
        analysis = {
            "contains_question": "?" in response,
            "length": len(response),
            "keywords": re.findall(r'\b\w+\b', response.lower())
        }
        return analysis