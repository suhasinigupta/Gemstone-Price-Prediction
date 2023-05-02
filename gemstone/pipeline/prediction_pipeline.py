import os, sys
from gemstone.logger import logging
from gemstone.exception import CustomException
from gemstone.utils import load_object
import pandas as pd

class PredictOutput :
    def __init__(self):
        pass

    def get_output(self, df):
        
      try:
         logging.info("Predicting the output and loading object ")
         preprocessor_obj= load_object(os.path.join('artifact','preprocessor.pkl'))
         model= load_object(os.path.join('artifact','model.pkl'))
    
         logging.info("Transforming the input dataframe ")
         df_transform= preprocessor_obj.transform(df)
         y_pred= model.predict(df_transform)

         return y_pred

      except Exception as e:
        raise CustomException(e, sys)

class CustomData:
    def __init__(self, carat:float,
                 depth:float,
                 table:float,
                 x:float,
                 y:float,
                 z:float,
                 cut:str,
                 color:str,
                 clarity:str):
        self.carat= carat
        self.depth=depth
        self.table=table
        self.x=x
        self.y=y
        self.z=z
        self.cut=cut
        self.color= color
        self.clarity=clarity

    def get_data_as_dataframe(self):
       try: 
           input_dict={'carat': self.carat, 'depth':self.depth, 'table':self.table, 'x':self.x, 'y':self.y, 'z':self.z,
                   'cut':self.cut, 'color':self.color, 'clarity': self.clarity}

           input_df= pd.DataFrame(input_dict)
           return input_df
       except Exception as e:
        raise CustomException(e, sys)