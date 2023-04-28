import os, sys
import pandas as pd
import numpy as np
from gemstone.logger import logging
from gemstone.exception import CustomException
from dataclasses import dataclass

@dataclass
class DataIngestionConfig:
    file_path :str= os.path.join('notebook', 'data', 'gemstones.csv')
    train_data:str= os.path.join('artifacts','train_data.csv')
    test_data:str= os.path.join('artifacts','test_data.csv')