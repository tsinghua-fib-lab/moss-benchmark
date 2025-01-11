import time
from functools import wraps

def timing_decorator(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs) 
        end_time = time.time()
        elapsed_time = end_time - start_time 
        
        # print(f"Function '{func.__name__}' took {elapsed_time:.9f} seconds to complete.")
        return result
    
    return wrapper
