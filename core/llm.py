from langchain_openai import ChatOpenAI
# from langchain.chat_models import init_chat_model
from langchain_google_genai import ChatGoogleGenerativeAI
from load_cfg import DEEPINFRA_API_KEY, DEEPINFRA_API_BASE
from logger import setup_logger
import os


class LanguageModelManager:
    def __init__(self):
        self.logger = setup_logger()
        self.llm = None
        self.power_llm = None
        self.json_llm = None
        self.initialize_llms()

    def initialize_llms(self):
        try:
            # self.llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", temperature=0)
            # self.power_llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro-latest", temperature=0.5)
            # self.json_llm = ChatGoogleGenerativeAI(
            #     model="gemini-1.5-pro-latest",
            #     temperature=0,
            #     generation_config={"response_mime_type": "application/json"}
            # )
            # self.logger.info("Google language models initialized successfully.")
            # meta-llama/Llama-3.3-70B-Instruct
            # Qwen/Qwen3-Next-80B-A3B-Instruct
            # Qwen3-235B-A22B-Thinking-2507
            # google/gemini-2.5-pro
            self.llm = ChatOpenAI(
                api_key=DEEPINFRA_API_KEY,
                base_url=DEEPINFRA_API_BASE,
                model="meta-llama/Llama-3.3-70B-Instruct-Turbo",
            )
            
            self.power_llm = ChatOpenAI(
                api_key=DEEPINFRA_API_KEY,
                base_url=DEEPINFRA_API_BASE,
                model="meta-llama/Llama-3.3-70B-Instruct-Turbo",
            )

            self.json_llm = ChatOpenAI(
                api_key=DEEPINFRA_API_KEY,
                base_url=DEEPINFRA_API_BASE,
                model="meta-llama/Llama-3.3-70B-Instruct-Turbo",
            )
            # self.llm = ChatOpenAI(
            #     # api_key=OPENAI_API_KEY,
            #     model="gpt-4o",
            #     temperature=0
            # )
            
            # self.power_llm = ChatOpenAI(
            #     # api_key=OPENAI_API_KEY,
            #     model="gpt-4o",
            #     temperature=0.5
            # )

            # self.json_llm = ChatOpenAI(
            #     # api_key=OPENAI_API_KEY,
            #     model="gpt-4o",
            #     temperature=0,
            #     model_kwargs={"response_format": {"type": "json_object"}}
            # )
            self.logger.info("DeepInfra language models initialized successfully.")
        except Exception as e:
            self.logger.error(f"Error creating LLM: {str(e)}")
            raise

    def get_models(self):
        return {
            "llm": self.llm,
            "power_llm": self.power_llm,
            "json_llm": self.json_llm
        }