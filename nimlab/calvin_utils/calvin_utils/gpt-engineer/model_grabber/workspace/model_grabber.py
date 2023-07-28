import inspect
import sys
from models import *

def model_grabber():
    while True:  # Keep running until the user accepts a model
        # Get all classes in models.py
        classes = inspect.getmembers(sys.modules[__name__], inspect.isclass)
        print("Available classes:")
        for name, _class in classes:
            print(f"- {name}")

        # Ask user to select a class
        class_name = input("Enter the name of the class to use: ")

        # Get the selected class and create an instance of it
        selected_class = getattr(sys.modules[__name__], class_name)
        selected_class_instance = selected_class()

        # Get the entry point function from the selected class
        entry_point_func = getattr(selected_class_instance, "start_function")

        # Get the input parameters of the entry point function
        args = inspect.signature(entry_point_func).parameters
        args_string = ', '.join(args.keys())

        # Ask user to specify input strings
        input_strings = {}
        for arg_name in args.keys():
            if arg_name == "self":  # Skip 'self', it's handled by the instance itself
                continue
            input_strings[arg_name] = input(f"Enter the value for {arg_name}: ")

        # Call the entry point function with the specified input strings
        result_string = entry_point_func(**input_strings)

        # Display the resulting string
        print(f'The model that will be used is: {result_string}')
        
        # Ask the user to accept or reject this model
        user_confirmation = input("Would you like to accept this model? (yes/no): ")
        if user_confirmation.lower() == "yes":
            # User accepted, return the resulting string
            return result_string