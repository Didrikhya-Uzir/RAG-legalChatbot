import os
import sys
from src.logger import logging
from src.exception import CustomException
from langchain_community.document_loaders import PyPDFLoader

from dataclasses import dataclass

@dataclass
class DataLoaderConfig:
    def __init__(self):
        self.file_path = os.path.join("artifacts","data","Indian Penal Code Book (2).pdf")

class DataLoader:
    def __init__(self):
        self.data_loader_obj = DataLoaderConfig()

    def pdf_loader(self):
        try:
            loader = PyPDFLoader(self.data_loader_obj.file_path)
            self.docs = loader.load()
            return self.docs
        
        except Exception as e:
            logging.info(f"Error occured in pdf_loader method. Error -- {e}")
            raise CustomException(e,sys)
    

if __name__ == "__main__":
    data_loader_obj = DataLoader()
    docs = data_loader_obj.pdf_loader()