import sys
import logging

def error_message_detail(error,error_detail:sys):
    """Whenever the error raise in my code this function will be called. This function gives on which file and which line no error occurs
    and what is the error  message we have"""
    _,_,exe_tb=error_detail.exc_info()
    filename = exe_tb.tb_frame.f_code.co_filename
    error_message =f"Error occured in python script {filename} and line no is {exe_tb.tb_lineno} and error message {str(error)}"
    
    return error_message 
class CustomException(Exception):
    def __init__(self,error_message,error_detail:sys):
        super().__init__(error_message)
        self.error_msg=error_message_detail(error=error_message,error_detail=error_detail)
        
        
    def __str__(self):
        return self.error_msg

    