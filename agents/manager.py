from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from core.llm import LanguageModelManager
from agents.insights_generator import InsightGeneratorAgent
from logger import setup_logger

logger = setup_logger()

class ManagerAgent:
    def __init__(self):
        self.logger = logger
        self.llm_manager = LanguageModelManager()
        self.llm = self.llm_manager.get_models()["llm"] 
        
        self.worker = InsightGeneratorAgent()

    def _check_ood_and_route(self, query: str) -> str:
        system_prompt = """
        You are the Manager of a Data Analysis System.
        Classify the user query into: 'DATA_ANALYSIS' or 'GENERAL'.
        Return ONLY the category name.
        """
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{query}")
        ])
        chain = prompt | self.llm | StrOutputParser()
        return chain.invoke({"query": query}).strip()

    def _analyze_requirements(self, query: str) -> str:
        system_prompt = """
        You are a Data Strategy Expert. Analyze the user's request and provide technical instructions for the Data Analyst.
        Identify:
        1. The goal (e.g., EDA, Feature Engineering, Modeling, Training, prediction, etc).
        2. Important constraints (e.g., "Check for missing values first", "Use Random Forest", etc).
        Output a concise instruction paragraph.
        """
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{query}")
        ])
        chain = prompt | self.llm | StrOutputParser()
        return chain.invoke({"query": query})

    def run(self, user_input: str):
        self.logger.info("Manager received input...")
        
        route_decision = self._check_ood_and_route(user_input)
        
        if route_decision == "DATA_ANALYSIS":
            self.logger.info("Routing to Insight Generator...")
            
            instructions = self._analyze_requirements(user_input)
            self.logger.info(f"Generated Instructions: {instructions}")
            
            return self.worker.run(task=user_input, specific_instructions=instructions)
        
        else:
            return self._handle_general_chat(user_input)

    def _handle_general_chat(self, query: str):
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful assistant. Guide user to provide data tasks."),
            ("human", "{query}")
        ])
        return (prompt | self.llm | StrOutputParser()).invoke({"query": query})