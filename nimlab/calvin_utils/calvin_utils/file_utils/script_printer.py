import argparse

class ScriptInfo:
    """
    This class is used to handle script information provided as a dictionary.
    
    Attributes:
        scripts: A list containing information about each script.

    Methods:
        print_script_name: Prints the name of the script.
        print_method: Prints the statistical method performed by the script.
        print_readme: Prints the README description of the script.
        print_inputs: Prints the required inputs for the script.
        print_high_level_info: Prints high level information for each script.
        print_info_by_method: Prints information for scripts matching a specified method.
    """
    def __init__(self, script_data):
        """
        Initialize ScriptInfo with data from the provided dictionary.

        Args:
            script_data (dict): A dictionary containing information about the scripts.
        """
        self.scripts = script_data['scripts']

    def print_script_name(self, script):
        """Prints the name of the script."""
        print(f"Script: {script['script_name']}")

    def print_method(self, script):
        """Prints the statistical method performed by the script."""
        print(f"Method: {script['method']}")

    def print_readme(self, script):
        """Prints the README description of the script."""
        print(f"README: {script['README']}")

    def get_docstring(self, script):
        """Prints the docstring of the imported script."""
        return script['get_docstring']

    def print_inputs(self, script):
        """
        Prints the required inputs for the script. Each input is printed with its corresponding description.
        """
        print("Inputs:")
        for input_name, description in script['inputs'].items():
            print(f"    {input_name}: {description}")

    def print_high_level_info(self):
        """
        Prints high level information about each script, including script name and method.
        """
        for script in self.scripts:
            self.print_script_name(script)
            self.print_method(script)
            print("---")
    
    def print_all_info(self):
        """
        Prints all information about each script, including script name, method, README, and inputs.
        """
        for script in self.scripts:
            self.print_script_name(script)
            self.print_method(script)
            self.print_readme(script)
            self.print_inputs(script)
            print("---")

    def print_inputs_by_method(self, method):
        """
        Prints all information about scripts that match a specified method.

        Args:
            method (str): The method to match.
        """
        for script in self.scripts:
            if script['method'] == method:
                self.print_script_name(script)
                self.print_method(script)
                self.print_inputs(script)
                print("---")
    
    def get_inputs_by_method(self, method):
        """
        Returns a dictionary of the required inputs for scripts that match a specified method.

        Args:
            method (str): The method to match.

        Returns:
            dict: A dictionary containing the required inputs for the matched scripts.
        """
        for script in self.scripts:
            if script['method'] == method:
                return script['inputs']
            
    def get_script_module_by_method(self, method):
        """
        Returns a dictionary of the module name for scripts that match a specified method.

        Args:
            method (str): The method to match.

        Returns:
            dict: A dictionary containing the module name for the matched scripts.
        """
        for script in self.scripts:
            if script['method'] == method:
                return 'from nimlab.calvin_utils.calvin_utils.statistical_utils.voxelwise_statistical_testing import '+script['script_name'].split('.py')[0]
        
    def get_module_by_method(self, method):
        """
        Returns a dictionary of the module name for scripts that match a specified method.

        Args:
            method (str): The method to match.

        Returns:
            dict: A dictionary containing the module name for the matched scripts.
        """
        for script in self.scripts:
            if script['method'] == method:
                return 'calvin_utils.permutation_analysis_utils.scripts_for_submission.'+script['script_name'].split('.py')[0]
    
    def get_script_import(self, method):
        """
        Returns the import.
        """
        for script in self.scripts:
            if script['method'] == method:
                return script['import']
            
    def get_docstring(self, method):
        """Prints the docstring of the imported script."""
        for script in self.scripts:
            if script['method'] == method:
                return script['get_docstring']
    def print_info_by_method(self, method):
        """
        Prints all information about scripts that match a specified method.

        Args:
            method (str): The method to match.
        """
        for script in self.scripts:
            if script['method'] == method:
                self.print_script_name(script)
                self.print_readme(script)
                self.print_inputs(script)
                print("---")
                
    def create_argparse_parser(self, script_name):
        """
        Creates an argparse ArgumentParser object for the specified script.

        Args:
            script_name (str): The name of the script.

        Returns:
            argparse.ArgumentParser: An ArgumentParser object with arguments defined
                based on the inputs for the specified script.
        """
        # Find the right dictionary in script_dict["scripts"] for the chosen script
        chosen_script_dict = next(script for script in self.scripts if script["script_name"] == script_name)
        
        # Create an ArgumentParser object
        parser = argparse.ArgumentParser(description=chosen_script_dict['README'])
        
        # Add arguments based on the "inputs" dictionary
        for input_name, description in chosen_script_dict['inputs'].items():
            parser.add_argument(f'--{input_name}', required=True, help=description)
        
        return parser
