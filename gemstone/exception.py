from gemstone.logger import logging
import os, sys


class CustomException:
    def __init__(self, error_message: Exception, error_detail: sys):
        super.__init__(error_message) 
        self.error_detail= error_detail
        self.error_messgae= CustomException.get_error_messgae(error_message, error_detail)

    @staticmethod
    def get_error_messgae(error_message, error_detail) -> str :
        _, _ , exc_tb= error_detail.exc_info()
        file_name = exc_tb.tb_frame.f_code.co_filename
        error_message = "Error occured in python script name [{0}] line number [{1}] error message [{2}]".format(
            file_name, exc_tb.tb_lineno, str(error_message)
        )

        return error_message
    
    def __str__(self):
       return self.error_message
    

    
