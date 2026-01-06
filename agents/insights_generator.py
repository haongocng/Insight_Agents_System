from langchain_core.messages import HumanMessage, SystemMessage
from core.llm import LanguageModelManager
from core.agent_factory import create_insight_worker_agent
from tools import (
    execute_code,
    execute_command,
    read_document,
    create_document,
    write_document,
    edit_document
)
from logger import setup_logger

logger = setup_logger()

class InsightGeneratorAgent:
    def __init__(self):
        self.logger = logger
        self.llm_manager = LanguageModelManager()
        
        self.tools = [
            execute_code,
            execute_command,
            read_document,
            create_document,
            write_document,
            edit_document
        ]
        
        self.agent_executor = create_insight_worker_agent(
            llm=self.llm_manager.get_models()["power_llm"],
            tools=self.tools
        )

    def run(self, task: str, specific_instructions: str = ""):
        try:
            self.logger.info(f"Insight Generator running task...")
            if specific_instructions:
                final_prompt = f"""
                CONTEXT & INSTRUCTIONS:
                {specific_instructions}
                
                USER TASK:
                {task}
                """
            else:
                final_prompt = task

            result = self.agent_executor.invoke(
                {"messages": [("human", final_prompt)]}
            )
            
            return result.get("output")
            
        except Exception as e:
            self.logger.error(f"Error in Insight Generator: {str(e)}")
            return f"Error executing task: {str(e)}"