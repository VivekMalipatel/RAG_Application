from typing import Dict, List, Optional
from datetime import datetime
from langchain.memory import ConversationBufferMemory
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

class TeachingSession:
    def __init__(self, student_id: str, subject: str, llm):
        self.student_id = student_id
        self.subject = subject
        self.llm = llm
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
        self.current_topic = None
        self.engagement_metrics = {
            "questions_asked": 0,
            "responses_given": 0,
            "comprehension_checks": []
        }
        self.session_start = datetime.now()
        self._initialize_teaching_chain()

    def _initialize_teaching_chain(self):
        """Initialize the teaching chain with appropriate prompts"""
        teaching_template = """
        You are an experienced and patient teacher. Your goal is to explain concepts 
        clearly and ensure student understanding. Use the following guidelines:
        
        - Break down complex topics into simple steps
        - Use analogies and real-world examples
        - Check comprehension regularly
        - Adjust explanations based on student responses
        
        Current Topic: {topic}
        Student's Prior Responses: {chat_history}
        Current Question/State: {current_input}
        
        Respond in a helpful, encouraging, and clear manner:
        """
        
        self.teaching_prompt = PromptTemplate(
            input_variables=["topic", "chat_history", "current_input"],
            template=teaching_template
        )
        
        self.teaching_chain = LLMChain(
            llm=self.llm,
            prompt=self.teaching_prompt,
            memory=self.memory
        )

    def start_topic(self, topic: str, content: str):
        """Start teaching a new topic"""
        self.current_topic = topic
        self.topic_start_time = datetime.now()
        # Here you would integrate with your friend's document processing pipeline
        # to get relevant content for the topic
        return self._generate_introduction(topic, content)

    def _generate_introduction(self, topic: str, content: str) -> str:
        """Generate an engaging introduction for the topic"""
        intro_prompt = f"""
        Based on the following content, create an engaging introduction for the topic '{topic}'.
        Make it relatable and interesting for the student.
        Content: {content}
        """
        return self.teaching_chain.run(
            topic=topic,
            current_input=intro_prompt,
            chat_history=self.memory.chat_memory.messages
        )

    def handle_student_input(self, student_input: str) -> str:
        """Process student input and generate appropriate response"""
        self.engagement_metrics["responses_given"] += 1
        
        if "?" in student_input:
            self.engagement_metrics["questions_asked"] += 1
        
        response = self.teaching_chain.run(
            topic=self.current_topic,
            current_input=student_input,
            chat_history=self.memory.chat_memory.messages
        )
        
        # Every few interactions, add a comprehension check
        if self.engagement_metrics["responses_given"] % 3 == 0:
            response += "\n\n" + self._generate_comprehension_check()
        
        return response

    def _generate_comprehension_check(self) -> str:
        """Generate a quick comprehension check question"""
        check_prompt = f"""
        Based on what we've discussed about {self.current_topic},
        generate a quick question to check understanding.
        Make it specific but not too complex.
        """
        return self.teaching_chain.run(
            topic=self.current_topic,
            current_input=check_prompt,
            chat_history=self.memory.chat_memory.messages
        )

    def get_session_metrics(self) -> Dict:
        """Get metrics for the current teaching session"""
        return {
            "session_duration": (datetime.now() - self.session_start).seconds / 60,
            "topics_covered": [self.current_topic],
            "engagement_metrics": self.engagement_metrics,
            "comprehension_level": self._calculate_comprehension_level()
        }

    def _calculate_comprehension_level(self) -> float:
        """Calculate estimated comprehension level based on interactions"""
        # This is a basic implementation - you can make it more sophisticated
        correct_responses = len([c for c in self.engagement_metrics["comprehension_checks"] if c["correct"]])
        total_checks = len(self.engagement_metrics["comprehension_checks"])
        return correct_responses / total_checks if total_checks > 0 else 0.0