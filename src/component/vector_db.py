import os
import sys
from src.logger import logging
from src.exception import CustomException
from dataclasses import dataclass
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
# from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.component.data_loader import DataLoader


@dataclass
class VectorDbConfig:
    def __int__(self):
        self.data_loader_obj = DataLoader()
        self.embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        docs = self.data_loader_obj.pdf_loader()
        self.splits = text_splitter.split_documents(docs)
        self.persist_directory = os.path.join("artifacts", "chroma_db")
class VectorDb:
    def __init__(self):
        self.VectorDbObj = VectorDbConfig()
        
    
    def VectorDbSave(self):
        try:
            self.vectorstore = Chroma.from_documents(
                documents= self.VectorDbObj.splits,
                embedding= self.VectorDbObj.embedding_model,
                persist_directory= self.VectorDbObj.persist_directory  # This automatically saves the DB
            )
            return self.vectorstore
        except Exception as e:
            logging.info(f"Error occured in VectorDbSave method. Error -- {e}")
            raise CustomException(e,sys)

    def VectorDbLoader(self):
        try:
            vectorstore = Chroma(
                persist_directory= self.VectorDbObj.persist_directory,
                embedding_function= self.VectorDbObj.embedding_model
            )
            self.retriever = vectorstore.as_retriever()
            return self.retriever
        except Exception as e:
            logging.info(f"Error occured in VectorDbLoader method. Error -- {e}")
            raise CustomException(e,sys)
    

if __name__ == "__main__":
    vector_db_obj = VectorDb()
    vector_store = vector_db_obj.VectorDbSave()
    retriever = vector_db_obj.VectorDbLoader()