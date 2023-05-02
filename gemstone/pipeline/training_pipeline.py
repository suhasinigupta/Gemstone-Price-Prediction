import os
import sys
from gemstone.logger import logging
from gemstone.exception import CustomException
import pandas as pd

from gemstone.components.data_ingestion import DataIngestion
from gemstone.components.data_transformation import DataTransformation
from gemstone.components.model_trainer import ModelTrainer


if __name__=='__main__':
    obj=DataIngestion()
    train_data_path,test_data_path=obj.initiate_data_ingestion()
    data_transformation = DataTransformation()
    train_arr,test_arr,_=data_transformation.initiate_data_transformation(train_data_path,test_data_path)
    model_trainer=ModelTrainer()
    model_trainer.initiate_model_training(train_arr,test_arr)