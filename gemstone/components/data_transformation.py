import os, sys
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from gemstone.logger import logging
from gemstone.exception import CustomException
from gemstone.utils import save_object
from dataclasses import dataclass

@dataclass
class DataTransformationConfig:
    preprocessor_file_path:str = os.path.join('artifact', 'preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.transformationconfig=DataTransformationConfig()

    def get_preprocessor_object(self):
        try:
           
           cat_cols = ['cut', 'color','clarity']
           num_cols = ['carat', 'depth','table', 'x', 'y', 'z']
           
           cut_categories = ['Fair', 'Good', 'Very Good','Premium','Ideal']
           color_categories = ['D', 'E', 'F', 'G', 'H', 'I', 'J']
           clarity_categories = ['I1','SI2','SI1','VS2','VS1','VVS2','VVS1','IF']

           num_pipeline= Pipeline(steps= [("Imputer", SimpleImputer(strategy="mean")), 
                               ("Standard Scaling", StandardScaler())])

           cat_pipeline= Pipeline(steps=[("Imputing",SimpleImputer(strategy ='most_frequent')),
                                        ("Ordinal Encoding", OrdinalEncoder(categories=[cut_categories,color_categories,clarity_categories])),
                                        ("Standard Scaling", StandardScaler())])

           preprocessor= ColumnTransformer([("num pipeline", num_pipeline, num_cols),
                                            ("cat pipelien", cat_pipeline, cat_cols)
            ])
           return preprocessor
        
        except Exception as e:
            logging.info("Exception occur in preprocessor stage")
            raise CustomException(e, sys)
        
    
    def initiate_data_transformation(self,train_data, test_data):
        try:

            logging.info("Starting Data Transformation")
            train_df= pd.read_csv(train_data)
            test_df= pd.read_csv(test_data)
            x_train= train_df.drop(columns=['price', 'id'], axis=1)
            x_test = test_df.drop(columns=['price','id'], axis=1) 
            
            y_train_df= train_df['price']
            y_test_df= test_df['price']
          
            preprocessor_obj= self.get_preprocessor_object()

            x_preprocessed_train= preprocessor_obj.fit_transform(x_train)
            x_preprocessed_test = preprocessor_obj.transform(x_test)

            train_arr = np.c_[x_preprocessed_train, np.array(y_train_df)]
            test_arr = np.c_[x_preprocessed_test, np.array(y_test_df)]

            save_object(

                file_path=self.transformationconfig.preprocessor_file_path,
                obj=preprocessor_obj

            )
            logging.info('Preprocessor pickle file saved')

            return (
                train_arr,
                test_arr,
                self.transformationconfig.preprocessor_file_path
            )
        except Exception as e:
            logging.info("Exception occur in initiate data transformation stage")
            raise CustomException(e, sys)