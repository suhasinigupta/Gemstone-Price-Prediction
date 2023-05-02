import os, sys
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
from sklearn.metrics import r2_score,  mean_absolute_error, mean_squared_error
from gemstone.logger import logging
from gemstone.exception import CustomException
from gemstone.utils import save_object
from gemstone.utils import evaluate_model
from dataclasses import dataclass

class ModelTrainerConfig:
    model_path: str= os.path.join('artifact', 'model.pkl')

class ModelTrainer:
    def __init__(self):
        self.modeltrainerconfig= ModelTrainerConfig()

    def initiate_model_training(self, train_array, test_array):
        try:
            logging.info('Splitting Dependent and Independent variables from train and test data')
            X_train, y_train, X_test, y_test = (
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )

            models={
            'LinearRegression':LinearRegression(),
            'Lasso':Lasso(),
            'Ridge':Ridge(),
            'Elasticnet':ElasticNet()
        }
            
            model_report:dict=evaluate_model(X_train,y_train,X_test,y_test,models)
            print(model_report)
            print('\n====================================================================================\n')
            logging.info(f'Model Report : {model_report}')

            # To get best model score from dictionary 
            best_model_score = max(sorted(model_report.values()))

            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            
            best_model = models[best_model_name]

            print(f'Best Model Found , Model Name : {best_model_name} , R2 Score : {best_model_score}')
            print('\n====================================================================================\n')
            logging.info(f'Best Model Found , Model Name : {best_model_name} , R2 Score : {best_model_score}')

            save_object(
                 file_path=self.modeltrainerconfig.model_path,
                 obj=best_model
            )
          

        except Exception as e:
            logging.info("Exception occur at Model Training Stage")
            raise CustomException(e, sys)