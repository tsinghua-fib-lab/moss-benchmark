import time
from functools import wraps

def timing_decorator(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()  # 记录开始时间
        result = func(*args, **kwargs)  # 调用原函数
        end_time = time.time()  # 记录结束时间
        elapsed_time = end_time - start_time  # 计算经过的时间
        
        print(f"Function '{func.__name__}' took {elapsed_time:.9f} seconds to complete.")
        return result
    
    return wrapper
