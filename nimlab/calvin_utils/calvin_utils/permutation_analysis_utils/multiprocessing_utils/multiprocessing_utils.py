#Perform analysis
from tqdm import tqdm
import os
import numpy as np
import pandas as pd
import concurrent.futures
from nimlab import datasets as nimds
from calvin_utils.permutation_analysis_utils.permutation_utils.palm import whole_brain_permutation_test
from calvin_utils.permutation_analysis_utils.permutation_utils.palm import permute_contrast_matrix
import inspect

import concurrent.futures
import importlib
from tqdm import tqdm
import inspect
import time
import time
import concurrent.futures


def collect_function(calvins_utils=True):
    # Ask the user for the module and function name
    if calvins_utils:
        module_name = input("Please enter the name of the python file to use within the calvin_utils directory (dont include extension): ")
        module_name = 'calvin_utils.' + module_name
    else:
        module_name = input("Please enter the name of the module with target function in this way (dont include extension): <directory_name.python_file_name> ")
    function_name = input("Please enter the name of the target function to be run: ")

    # Feedback to user
    print('Checking for module: ', module_name)
    print('Checking in module for function: ', function_name)
    
    # Import the function
    try:
        module = importlib.import_module(module_name)
        function_to_run = getattr(module, function_name)
    except (ModuleNotFoundError, AttributeError) as e:
        print(f"Error: {e}")
    return function_to_run

def collect_args(func):
    # Get the function signature and parameters
    signature = inspect.signature(func)
    parameters = signature.parameters
 
    # Print the docstring if present
    if func.__doc__:
        print("Function documentation:")
        print(func.__doc__)
        print()

    # Collect user input based on the function parameters
    args = {}
    for name, param in parameters.items():
        # Check if the parameter has a default value
        if param.default == inspect.Parameter.empty:
            user_input = input(f"Please enter a value for '{name}': ")
        else:
            user_input = input(f"Please enter a value for '{name}' (default: {param.default}): ")
        
        # Convert the input to the correct type
        user_input = param.annotation(user_input) if param.annotation != inspect.Parameter.empty else user_input
        args[name] = user_input

    return args

def collect_user_and_job_info():
    # Collect user information
    name = input("Please enter your name: ")
    email = input("Please enter your email: ")
    cores = int(input("Please enter the number of cores that you can user per job (5 is a safe bet): "))
    n_permutations = int(input('How many permutations would you like to perform: '))
    job_name = input('Please enter a job name for this task: ')
    data_path = input('Please enter absolute path to the csv which contains information to be permuted: ')
    out_dir = input('Please enter absolute path to output directory: ')

    # Validate the number of cores
    while cores < 1 or cores > 64:
        print("Invalid number of cores. Please enter a value between 1 and 64.")
        cores = int(input("Please enter the number of cores (1-64): "))

    # Return collected info as a dictionary
    return {
        "name": name,
        "email": email,
        "cores": cores,
        "n_permutations": n_permutations,
        "job_name": job_name,
        "data_path": data_path,
        "out_dir": out_dir
    }

# Initialize the matrix to store the output results
def submit_job_to_cpu_set(func, args, matrix_to_compute, n_cores=5):
    empiric_p_matrix = np.zeros_like(matrix_to_compute)
    with concurrent.futures.ProcessPoolExecutor(max_workers=n_cores) as executor:
        #Begin submitting the masked data to the permutor
        results = []
        for i in tqdm(range(n_cores), desc="Jobs Launched"):
            ### Some -function- to be run i=n_cores times
            result = executor.submit(func, **args)
            results.append(result)

        progress_bar = tqdm(total=n_cores, desc="Jobs Finalized")
        for result in concurrent.futures.as_completed(results):
            #Input the permuted data into the array
            extracted_p_count = result.result()
            empiric_p_matrix = empiric_p_matrix + extracted_p_count
            
            #Update visualization
            progress_bar.update()
        progress_bar.close()
    return empiric_p_matrix

def submit_jobs_in_batches(function_to_submit, function_args, out_dir, job_name, n_permutations=10000, n_cores=5, delay=0.5):
    # Calculate the number of batches
    n_batches = n_permutations // n_cores

    # Initialize an empty dataframe to store the cumulative results
    cumulative_p_counts = pd.DataFrame()

    # Create a ThreadPoolExecutor to run the submit_job_to_cpu_set function asynchronously
    with concurrent.futures.ThreadPoolExecutor() as executor:
        # Create a list to store the future objects
        futures = []

        # Loop through the batches
        for batch_idx in range(n_batches):
            # Submit the job and store the future object
            future = executor.submit(submit_job_to_cpu_set, n_cores=n_cores, function_to_submit=function_to_submit, function_args=function_args)
            futures.append(future)

            # Introduce a delay between job submissions to avoid overloading the scheduler
            if batch_idx < n_batches - 1:
                time.sleep(delay)

        # Collect the results from the completed futures
        progress_bar = tqdm(total=n_permutations, desc="Jobs Finalized")
        for future in concurrent.futures.as_completed(futures):
            p_count_per_batch_df = future.result()

            # Add the returned dataframe from the current batch to the cumulative dataframe
            if cumulative_p_counts.empty:
                cumulative_p_counts = p_count_per_batch_df
            else:
                cumulative_p_counts += p_count_per_batch_df
            #Update visualization
            progress_bar.update()
        progress_bar.close()
    
    #Calculate the p-value from the empiric distribution    
    p_values_df = cumulative_p_counts / n_permutations
    
    # Save the cumulative results to a file
    output_file = os.path.join(out_dir, f'{job_name}.csv')
    p_values_df.to_csv(output_file, index=False)
