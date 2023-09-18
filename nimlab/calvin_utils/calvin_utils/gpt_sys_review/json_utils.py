from fuzzywuzzy import fuzz
import pandas as pd
import numpy as np
import json
import os

class FilterPapers:
    '''
    The FilterPapers class provides functionality to filter a dataset of articles based on inclusion/exclusion criteria.
    
    This class takes the paths to a CSV file and a JSON file as input. The CSV file should contain the articles that have passed the inclusion/exclusion criteria. The JSON file should contain the labeled sections of all the articles under consideration.
    
    The class offers methods to:
    1. Read the CSV and JSON files.
    2. Filter the JSON data based on the articles listed in the CSV.
    3. Save the filtered JSON data to a new JSON file.
    
    Attributes:
    - csv_path (str): Path to the CSV file containing articles that passed the inclusion/exclusion criteria.
    - json_path (str): Path to the JSON file containing labeled sections of all articles.
    - df (DataFrame): DataFrame containing articles that passed the inclusion/exclusion criteria.
    - data (dict): Dictionary containing the labeled sections of all articles.
    
    Example:
    csv_path = "/mnt/data/sample_articles.csv"
    json_path = "/mnt/data/labeled_sections.json"
    filter_papers = FilterPapers(csv_path=csv_path, json_path=json_path)
    filter_papers.run()
    '''
    def __init__(self, csv_path, json_path):
        """
        Initializes the FilterPapers class with paths to the CSV and JSON files.
        
        Parameters:
        - csv_path (str): Path to the CSV file containing the articles that passed the inclusion/exclusion criteria.
        - json_path (str): Path to the JSON file containing the labeled sections of all articles.
        """
        self.csv_path = csv_path
        self.json_path = json_path
        self.df = self.read_csv()
        self.data = self.read_json()

    def read_csv(self):
        """
        Reads the CSV file into a DataFrame. Assumes the first column of the CSV is the index.
        
        Returns:
        - DataFrame: DataFrame containing the articles that passed the inclusion/exclusion criteria.
        """
        return pd.read_csv(self.csv_path, index_col=0)

    def read_json(self):
        """
        Reads the JSON file into a dictionary.
        
        Returns:
        - dict: Dictionary containing the labeled sections of all articles.
        """
        with open(self.json_path, 'r') as file:
            return json.load(file)

    def filter_json(self):
        """
        Filters the JSON data based on the DataFrame. It selects only those articles that are present in the DataFrame's index.
        
        Returns:
        - dict: Dictionary containing the labeled sections of articles that passed the inclusion/exclusion criteria.
        """
        filtered_data = {key: value for key, value in self.data.items() if key in self.df.index}
        return filtered_data

    def save_to_json(self):
        """
        Saves the filtered JSON data to a new file in a directory called `filtered_articles`.
        
        Returns:
        - None
        """
        # Create a new directory in the same root folder
        out_dir = os.path.join(os.path.dirname(self.json_path), "filtered_articles")
        os.makedirs(out_dir, exist_ok=True)
        
        # Save the filtered dictionary to a JSON file
        with open(os.path.join(out_dir, 'filtered_labeled_sections.json'), 'w') as f:
            json.dump(self.filter_json(), f, indent=4)

    def run(self):
        """
        A convenience method that calls `save_to_json()` to execute the filtering and saving in one step.
        
        Returns:
        - None
        """
        self.save_to_json()

class InclusionExclusionSummarizer:
    """
    Class to summarize inclusion/exclusion criteria based on the answers received from GPT-3.5.
    
    Attributes:
    - json_path (str): Path to the JSON file containing the answers.
    - data (dict): The data read from the JSON file.
    - df (DataFrame): Pandas DataFrame to store summarized results.
    """
    
    def __init__(self, json_path):
        """
        Initializes the InclusionExclusionSummarizer class.
        
        Parameters:
        - json_path (str): Path to the JSON file containing the answers.
        """
        self.json_path = json_path
        self.data = self.read_json()
        self.df = self.summarize_results()
    
    def read_json(self):
        """
        Reads JSON data from a file.
        
        Returns:
        - dict: The data read from the JSON file.
        """
        with open(self.json_path, 'r') as file:
            return json.load(file)
    
    def summarize_results(self):
        """
        Summarizes the results by converting answers to binary form.
        
        Returns:
        - DataFrame: Pandas DataFrame containing the summarized results.
        """
        summary_dict = {}
        for article, questions in self.data.items():
            summary_dict[article] = {}
            for question, chunks in questions.items():
                # Convert all chunk answers to lowercase and check for "yes" keywords
                binary_answers = [1 if 'y' in answer.lower() else 0 for answer in chunks.values()]
                # Sum up the binary answers for each question
                summary_dict[article][question] = sum(binary_answers)
        
        # Convert the summary dictionary to a DataFrame
        df = pd.DataFrame.from_dict(summary_dict, orient='index')
        
        # Set all values above 0 to 1
        df[df > 0] = 1
        
        return df
    
    def drop_rows_with_zeros(self):
        """
        Drops any row in the DataFrame that contains a zero.
        
        Returns:
        - DataFrame: A new DataFrame with rows containing zeros removed.
        """
        return self.df[(self.df == 0).sum(axis=1) == 0]
    
    def save_to_csv(self, dropped=False, filename='inclusion_exclusion_results'):
        """
        Saves the DataFrame to a CSV file.
        
        Parameters:
        - dropped (bool): Indicates whether rows have been dropped from the DataFrame.
        
        Returns:
        - None
        """
        # Create a new directory in the same root folder
        out_dir = os.path.join(os.path.dirname(self.json_path), filename)
        os.makedirs(out_dir, exist_ok=True)
        
        # Determine the name of the CSV file based on whether rows have been dropped
        file_name = "automated_filtered_results.csv" if dropped else "raw_results.csv"
        
        # Save the DataFrame to a CSV file
        csv_path = os.path.join(out_dir, file_name)
        if dropped:
            self.drop_rows_with_zeros().to_csv(csv_path)
        else:
            self.df.to_csv(csv_path)
            
    def run(self):
        """
        Executes all the summarization, saving and optional row-dropping steps in one method.
        
        Returns:
        - None
        """
        self.save_to_csv()
        self.save_to_csv(dropped=True)
        return self.df

class CustomSummarizer:
    """
    Custom class to summarize research evaluation data based on fuzzy matching and keyword mapping.
    
    Attributes:
    - json_path (str): Path to the JSON file containing answers.
    - keyword_mapping (dict): Mapping of keywords to values.
    - data (dict): Data read from the JSON file.
    - df (DataFrame): DataFrame to store summarized results.
    """
    def __init__(self, json_path, keyword_mapping):
        """
        Initializes the CustomSummarizerFinal class.
        
        Parameters:
        - json_path (str): Path to the JSON file containing answers.
        - keyword_mapping (dict): Mapping of keywords to values.
        """
        import json
        self.json_path = json_path
        self.keyword_mapping = keyword_mapping
        self.data = self.read_json()
        self.df = self.summarize_results_with_mapping()
    
    def read_json(self):
        """
        Reads data from the JSON file specified during initialization.
        
        Returns:
        - dict: Data read from the JSON file.
        """
        with open(self.json_path, 'r') as file:
            return json.load(file)
    
    def exact_match(self, answer):
        """
        Checks for an exact match of keywords in the answer text.
        
        Parameters:
        - answer (str): The answer text to be matched.
        
        Returns:
        - int or np.nan or None: Returns the key if a match is found, otherwise None.
        """
        import re
        cleaned_answer = re.sub(r'[^\w\s]', '', answer.lower())
        for key, keywords in self.keyword_mapping.items():
            for keyword in keywords:
                if keyword.lower() in cleaned_answer.split():
                    return key
        return None
    
    def fuzzy_match(self, answer, threshold=60):
        """
        Fuzzy matches the answer with a list of keywords.
        
        Parameters:
        - answer (str): The answer text to be matched.
        - threshold (int): The similarity ratio threshold for a valid match.
        
        Returns:
        - int or np.nan or None: Returns the key if a match is found, otherwise None.
        """
        from fuzzywuzzy import fuzz
        best_match = None
        highest_ratio = 0
        for key, keywords in self.keyword_mapping.items():
            for keyword in keywords:
                ratio = fuzz.ratio(answer.lower(), keyword.lower())
                if ratio > highest_ratio:
                    highest_ratio = ratio
                    best_match = key
        if highest_ratio >= threshold:
            return best_match
        else:
            return None
    
    def keyword_or_fuzzy_match(self, answer):
        """
        Applies either exact matching or fuzzy matching based on the result of exact matching.
        
        Parameters:
        - answer (str): The answer text to be matched.
        
        Returns:
        - int or np.nan or None: Returns the key if a match is found, otherwise None.
        """
        exact_result = self.exact_match(answer)
        return exact_result if exact_result is not None else self.fuzzy_match(answer)
    
    def summarize_results_with_mapping(self):
        """
        Summarizes the results based on keyword mapping and fuzzy matching.
        
        Returns:
        - DataFrame: Pandas DataFrame containing the summarized results.
        """
        import numpy as np
        import pandas as pd
        summary_dict = {}
        for article, questions in self.data.items():
            summary_dict[article] = {}
            for question, chunks in questions.items():
                if self.keyword_mapping:
                    mapped_answers = [self.keyword_or_fuzzy_match(answer) for answer in chunks.values()]
                    if all(x is np.nan for x in mapped_answers) or all(x is None for x in mapped_answers):
                        summary_dict[article][question] = np.nan
                    else:
                        valid_answers = [x for x in mapped_answers if x is not np.nan and x is not None]
                        summary_dict[article][question] = np.sum(valid_answers) if valid_answers else np.nan
        df = pd.DataFrame.from_dict(summary_dict, orient='index').fillna(np.nan).astype('Int64')
        return df
    
class CustomSummarizer(InclusionExclusionSummarizer):
    """
    Class to create a custom summary of the results with user-defined keyword mapping and fuzzy matching.
    
    Attributes:
    - keyword_mapping (dict): Dictionary mapping each key to a list of acceptable values.
    """
    
    def __init__(self, json_path, keyword_mapping=None):
        """
        Initializes the CustomSummarizer class.
        
        Parameters:
        - json_path (str): Path to the JSON file containing the answers.
        - keyword_mapping (dict): Dictionary mapping each key to a list of acceptable values.
        """
        super().__init__(json_path)
        self.keyword_mapping = keyword_mapping
    
    def exact_match(self, answer):
        """
        Checks for an exact match of keywords in the answer text.
        
        Parameters:
        - answer (str): The answer text to be matched.
        
        Returns:
        - int or np.nan or None: Returns the key if a match is found, otherwise None.
        """
        import re
        cleaned_answer = re.sub(r'[^\w\s]', '', answer.lower())
        for key, keywords in self.keyword_mapping.items():
            for keyword in keywords:
                if keyword.lower() in cleaned_answer.split():
                    return key
        return None
    
    def fuzzy_match(self, answer, threshold=60):
        """
        Fuzzy matches the answer with a list of keywords.
        
        Parameters:
        - answer (str): The answer text to be matched.
        - threshold (int): The similarity ratio threshold for a valid match.
        
        Returns:
        - int or np.nan or None: Returns the key if a match is found, otherwise None.
        """
        from fuzzywuzzy import fuzz
        best_match = None
        highest_ratio = 0
        for key, keywords in self.keyword_mapping.items():
            for keyword in keywords:
                ratio = fuzz.ratio(answer.lower(), keyword.lower())
                if ratio > highest_ratio:
                    highest_ratio = ratio
                    best_match = key
        if highest_ratio >= threshold:
            return best_match
        else:
            return None
    
    def keyword_or_fuzzy_match(self, answer):
        """
        Applies either exact matching or fuzzy matching based on the result of exact matching.
        
        Parameters:
        - answer (str): The answer text to be matched.
        
        Returns:
        - int or np.nan or None: Returns the key if a match is found, otherwise None.
        """
        exact_result = self.exact_match(answer)
        return exact_result if exact_result is not None else self.fuzzy_match(answer)
    
    def summarize_results_with_mapping(self):
        """
        Summarizes the results based on keyword mapping and fuzzy matching.
        
        Returns:
        - DataFrame: Pandas DataFrame containing the summarized results.
        """
        import numpy as np
        import pandas as pd
        summary_dict = {}
        for article, questions in self.data.items():
            summary_dict[article] = {}
            for question, chunks in questions.items():
                if self.keyword_mapping:
                    mapped_answers = [self.keyword_or_fuzzy_match(answer) for answer in chunks.values()]
                    if all(x is np.nan for x in mapped_answers) or all(x is None for x in mapped_answers):
                        summary_dict[article][question] = np.nan
                    else:
                        valid_answers = [x for x in mapped_answers if x is not np.nan and x is not None]
                        summary_dict[article][question] = np.sum(valid_answers) if valid_answers else 'Unidentified'
        df = pd.DataFrame.from_dict(summary_dict, orient='index').fillna(np.nan)
        return df
    
    def run_custom(self):
        """
        Executes all the summarization, saving, and optional row-dropping steps in one method.
        
        Returns:
        - DataFrame: Pandas DataFrame containing the summarized results.
        """
        self.df = self.summarize_results_with_mapping()
        self.save_to_csv(filename='data_extraction_results')
        self.save_to_csv(dropped=True, filename='data_extraction_results')
        return self.df