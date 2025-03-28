import os
import sys
from langchain_mistralai import ChatMistralAI
# from langchain_core.messages import AIMessage
from dotenv import load_dotenv
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dataclasses import dataclass
from src.component.vector_db import VectorDb
from src.logger import logging
from src.exception import CustomException

@dataclass
class PredictionMistralConfig:
    model="mistral-large-latest"
    # vector_db_path = os.path.join("artifacts","prepare_vector_db")
    temprature = 0.2
    # embedding_model_name = "BAAI/bge-base-en"
    model_kwargs = {"device": "cpu"}
    encode_kwargs = {"normalize_embeddings": True}

class PredictionMistral:
    def __init__(self):
        self.prediction_config = PredictionMistralConfig()
        self.vector_db_obj = VectorDb()
        self.retriever = self.vector_db_obj.VectorDbLoader()

    def load_model(self):
        try:
            load_dotenv()
            api_key = os.getenv("MISTRAL_API_KEY")
            os.environ["MISTRAL_API_KEY"] = api_key
            self.llm = ChatMistralAI(model=self.prediction_config.model,
                                     temprature = self.prediction_config.temprature,
                                     max_retries = 2)
            print("*"*100)
            print("model load successfully")
        except Exception as e:
            logging.info(f"Error in load_model -- {e}")
            raise CustomException(e,sys)
    
    def querry_response(self, message):
        try:
            system_prompt = (
                """You are a trained bot to guide people about Indian Law. You will answer user's query with your knowledge and the context provided.
                        If a question does not make any sense, or is not factually coherent, explain why instead of answering something incorrect. If you don't know the answer to a question, please don't share false information.
                        Do not say thank you and tell you are an AI Assistant. Be open about everything.
                        Use the following pieces of context to answer the user's question.
                        Context: {context}
                        Only return the helpful answer below and nothing else.
                        Helpful answer:"""
            )

            prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", system_prompt),
                    ("human", "{input}"),
                ]
            )
            question_answer_chain = create_stuff_documents_chain(self.llm, prompt)
            self.rag_chain = create_retrieval_chain(self.retriever, question_answer_chain)
            self.results = self.rag_chain.invoke({"input": f"{message}"})
            return self.results["answer"]
        
        except Exception as e:
            logging.info(f"Error in querry_response -- {e}")
            raise CustomException(e,sys)


if __name__ == "__main__":
    obj = PredictionMistral()
    obj.load_model()
    obj.querry_response(message="Explain me about the document?")