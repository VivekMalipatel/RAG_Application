from typing import Dict, List, Optional
from datetime import datetime
from langchain.memory import ConversationBufferMemory
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from .metrics import EngagementMetrics
from .strategies import TeachingStrategy
from ..utils.logger import setup_logger
from ..utils.config import load_config

logger = setup_logger(__name__)

class TeachingSession:
    def __init__(self, student_id: str, subject: str, llm):
        self.student_id = student_id
        self.subject = subject
        self.llm = llm
        self.config = load_config()
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
        self.metrics = EngagementMetrics()
        self.current_topic = None
        self._initialize_teaching_chain()
        logger.info(f"Teaching session initialized for student {student_id}")

    def _initialize_teaching_chain(self):
        """Initialize the teaching chain with GPT-optimized prompts"""
        teaching_template = """
        You are an experienced and patient teacher. Your role is to help students understand concepts clearly and build their confidence. 

        Guidelines:
        1. Break complex topics into digestible steps
        2. Use clear, relatable examples
        3. Keep responses concise but informative
        4. Be encouraging and supportive
        5. Check understanding regularly
        6. Adapt explanations based on student responses

        Previous Discussion:
        {chat_history}

        Current Topic: {topic}
        Student's Question/Input: {current_input}

        Respond as a helpful teacher would, maintaining a conversational and encouraging tone:
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

    def start_topic(self, topic: str, content: str) -> str:
        """Start teaching a new topic"""
        self.current_topic = topic
        logger.info(f"Starting new topic: {topic}")
        return TeachingStrategy.create_topic_introduction(topic, content)

    def handle_student_input(self, student_input: str) -> str:
        """Process student input and generate appropriate response"""
        # Record metrics
        self.metrics.record_response()
        
        # Analyze student response
        analysis = TeachingStrategy.analyze_student_response(student_input)
        if analysis["contains_question"]:
            self.metrics.record_question()
        
        # Generate response
        response = self.teaching_chain.run(
            topic=self.current_topic,
            current_input=student_input,
            chat_history=self.memory.chat_memory.messages
        )
        
        # Add comprehension check if needed
        if self.should_check_comprehension():
            response += "\n\n" + TeachingStrategy.generate_comprehension_check(
                self.current_topic,
                student_input
            )
        
        return response

    def should_check_comprehension(self) -> bool:
        """Determine if we should perform a comprehension check"""
        responses = self.metrics.metrics["responses_given"]
        interval = self.config["teaching"]["comprehension_check_interval"]
        return responses > 0 and responses % interval == 0

    def get_session_metrics(self) -> Dict:
        """Get current session metrics"""
        metrics = self.metrics.get_metrics()
        metrics["session_duration"] = (datetime.now() - metrics["session_start"]).seconds
        return metrics

class SessionManager:
    def __init__(self):
        self.active_sessions: Dict[str, TeachingSession] = {}
        self.logger = setup_logger(__name__ + ".SessionManager")
    
    def create_session(self, student_id: str, subject: str, llm) -> TeachingSession:
        """Create a new teaching session"""
        session = TeachingSession(student_id, subject, llm)
        self.active_sessions[student_id] = session
        self.logger.info(f"Created new session for student {student_id}")
        return session
    
    def get_session(self, student_id: str) -> Optional[TeachingSession]:
        """Retrieve an existing session"""
        return self.active_sessions.get(student_id)
    
    def end_session(self, student_id: str):
        """End a teaching session"""
        if student_id in self.active_sessions:
            del self.active_sessions[student_id]
            self.logger.info(f"Ended session for student {student_id}")