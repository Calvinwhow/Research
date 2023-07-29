"""
Module: memory_management.py

This module provides a class, MemoryCheckingExecutor, which is designed to manage memory usage 
in the context of concurrent task execution. It uses the built-in concurrent.futures module 
to execute tasks concurrently, while also monitoring the system's memory usage to ensure that 
it does not exceed a user-specified threshold. 

This is particularly useful when executing a large number of memory-intensive tasks, such as 
large data processing or scientific computation tasks, as it allows for efficient use of 
computational resources while also preventing out-of-memory errors.

Classes:
    - MemoryCheckingExecutor

"""

import concurrent.futures
import psutil
import time
import numpy as np


class MemoryCheckingExecutor:
    """
    MemoryCheckingExecutor is a class that manages the execution of tasks in a way that memory usage is kept under a 
    specified threshold. It is initialized with the maximum number of concurrent tasks it should run (max_workers), the 
    maximum memory usage allowed (threshold_memory_gb), and an estimate of the maximum memory a single task might use 
    (task_memory_gb).

    Methods:
    - submit(function, *args, **kwargs): Submit a task to the executor. The task will only be started if the estimated 
      memory usage after starting the task will be below the threshold.
    - task_done(future): Callback for when a task is done. Decrements the task counter.
    - memory_usage_over_limit(): Checks if the estimated memory usage is over the limit. 

    """
    def __init__(self, max_workers, task_memory_gb):
        """
        Initialize the MemoryCheckingExecutor.

        Parameters:
        - max_workers: The maximum number of tasks to execute concurrently.
        - task_memory_gb: The estimated maximum memory usage of a single task in gigabytes. 
        """
        self.executor = concurrent.futures.ProcessPoolExecutor(max_workers=max_workers)
        self.task_memory_gb = task_memory_gb
        self.threshold_memory_gb = int(np.round(self.task_memory_gb*.75))
        self.current_tasks = 0
        
    def __enter__(self):
        """
        Enter the runtime context related to this object. The with statement will bind this method's return value
        to the target(s) specified in the as clause of the statement, if any.

        Returns:
        - self: Returns the instance of the MemoryCheckingExecutor itself. 
        """
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Exit the runtime context related to this object and performs any necessary cleanup.

        Parameters:
        - exc_type: The exception type, if an exception was raised in the context. None otherwise.
        - exc_val: The exception instance, if an exception was raised in the context. None otherwise.
        - exc_tb: A traceback object encoding the details of the call stack at the point where the exception was raised, if an exception was raised in the context. None otherwise.

        Note:
        - This method should not re-raise any exceptions that occur within it. It should return False to propagate exceptions, or True to suppress them.
        """
        self.executor.shutdown()
        return False

    def submit(self, function, *args, **kwargs):
        """
        Submit a task to be executed. The task will only be started if the estimated memory usage, including the 
        memory required for this task, is below the threshold.

        Parameters:
        - function: The function to be executed.
        - *args, **kwargs: Arguments and keyword arguments to be passed to the function.

        Returns:
        - A concurrent.futures.Future representing the execution of the task.
        """
        while self.memory_usage_over_limit():
            print("Memory usage is over the limit. Waiting for 10 seconds before checking again.")
            time.sleep(10)

        self.current_tasks += 1
        future = self.executor.submit(function, *args, **kwargs)
        future.add_done_callback(self.task_done)
        return future

    def task_done(self, future):
        """
        Callback for when a task is done. Decrements the task counter.

        Parameters:
        - future: The concurrent.futures.Future representing the task.
        """
        self.current_tasks -= 1

    def memory_usage_over_limit(self):
        """
        Check if the projected memory usage is over the limit. The projected memory usage is the sum of the currently 
        used memory and the estimated memory usage of the running tasks.

        Returns:
        - True if the projected memory usage is over the limit, False otherwise.
        """
        memory = psutil.virtual_memory()
        used_memory_gb = memory.used / (1024 ** 3)
        projected_memory_gb = used_memory_gb + self.task_memory_gb * self.current_tasks
        return projected_memory_gb >= self.threshold_memory_gb
