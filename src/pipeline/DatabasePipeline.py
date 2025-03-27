import os
import sys
from dataclasses import dataclass
from src.component.vector_db import VectorDb
from src.logger import logging
from src.exception import CustomException

@dataclass
class CommandPipeline:
    def __init__(self):
        pass

    def initiate_pipeline(self):
        try:
            VectorDb_obj = VectorDb()
            VectorDb_obj.VectorDbSave()
            print("Database pipeline executed successfully.")
            return None

        except Exception as e:
            logging.info(f"Error in command pipeline ---- {e}")
            raise CustomException(e,sys)


# if __name__ == "__main__":
#     pipeline = CommandPipeline()
#     pipeline.initiate_pipeline(command="xyz")