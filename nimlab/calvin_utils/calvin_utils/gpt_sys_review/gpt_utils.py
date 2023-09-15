import openai
import json
import os 
import time
import numpy as np
from tqdm import tqdm
from calvin_utils.gpt_sys_review.txt_utils import TextChunker
    
class QuestionTemplate:
    """
    Class to manage and display templates for various types of questions.
    
    # Example usage
    question_template = QuestionTemplate()
    question_template.inclusion_exclusion_questions()
    question_template.data_extraction_questions()
    question_template.print_question_template("inclusion")
    question_template.print_question_template("exclusion")
    question_template.print_question_template("custom")
    """
    def __init__(self):
        self.get_question_templates(self)
    
    def inclusion_exclusion_questions(self):
        """Method to print inclusion and exclusion questions."""

        print("Here are example inclusion questions:")
        print(json.dumps(self.inclusion_questions, indent=4))
        print("Here are example exclusion questions:")
        print(json.dumps(self.exclusion_questions, indent=4))
        print("Here is a question template")
        print(json.dumps(self.template_questions, indent=4))
    
    def data_extraction_questions(self):
        """Method to print data extraction questions."""
        
        print("Here are example data extraction questions:")
        print(json.dumps(self.strict_extraction_questions, indent=4))
        print("Here are some more generalizable data extraction questions:")
        print(json.dumps(self.lenient_extraction_questions, indent=4))
        print("Here is a question template")
        print(json.dumps(self.template_questions, indent=4))
    
    def get_question_templates(self, question_type):
        """
        Prints out a template for questions based on the specified question type.
        
        Parameters:
        - question_type (str): Type of questions to print ('inclusion', 'exclusion', 'evaluation').
        
        Returns:
        None
        """
        self.inclusion_questions = {'Amnesia case report? (Y/N)': 'case_report',
                            'Published in English? (Y/N)': 'is_english'}
        self.exclusion_questions = {'Transient amnesia, reversible amnesia symptom, severe confabulation or drug use, toxicity, epilepsy-related confusion, psychological or psychiatric-related amnesia (functional amnesia)': 'other_cause',
                            'Did not examine/report both retrograde and anterograde memory domains': 'not_both_domains',
                            'Without descriptive/qualitative/quantitative data on amnesia severity/memory tests/questions/scenarios/details': 'not_enough_information',
                            'Had global cognitive impairment disproportionate to memory loss': 'disproportionate_impairment',
                            'Without measurable lesion-related brain MR/CT scans': 'no_scan',
                            'Had focal or widespread brain atrophy': 'neurodegenerative',
                            'Atypical cases with selective (e.g., semantic) memory loss or material/topographic-specific memory loss': 'atypical_case'
                        }  
        self.strict_extraction_questions = {'Does the patient(s) represent(s) the whole experience of the investigator (center) or is the selection method unclear to the extent that other patients with similar presentation may not have been reported? (Good/Bad/Unclear)': 'representative_case_quality',
                            'Was patient’s causal exposure clearly described? (Good/Bad/Unclear)': 'causality_quality',
                            'Were diagnostic tests or assessment methods and the results clearly described (amnesia tests)? (Good/Bad/Unclear)': 'phenotyping_quality',
                            'Were other alternative causes that may explain the observation (amnesia) ruled out? (Good/Bad/Unclear)': 'workup_quality',
                            'Were patient’s demographics, medical history, comobidities clearly described? (Good/Bad/Unclear)': 'clinical_covariates_quality',
                            'Were patient’s symptoms, interventions, and clinical outcomes clearly presented as a timeline? (Good/Bad/Unclear)': 'history_quality',
                            'Was the lesion image taken around the time of observation (amnesia) assessment? (Good/Bad/Unclear)': 'temporal_causality_quality',
                            'Is the case(s) described with sufficient details to allow other investigators to replicate the research or to allow practitioners make inferences related to their own practice? (Good/Bad/Unclear)': 'history_quality_2'
            }
        self.lenient_extraction_questions = {'How affected is this study by selection bias? Little (Good). Very (Bad). Unsure (Unclear).': 'selection_bias',
                            'How well was the timeline of lesion/symptom onset described? (Good/Bad/Unclear)': 'exposure',
                            'How affected is this case by attribution error? Little (Good). Very (Bad). Unsure (Unclear).': 'outcome',
                            'How reasonable is the attribution of the lesion/diagnosis to the symptom? (Good/Bad/Unclear)': 'attribution_error',
                            'How well was the patients baseline pre-lesion described? (Good/Bad/Unclear)': 'pre_event_assessment',
                            'How well was the patients outcome post-lesion described? (Good/Bad/Unclear)': 'post_event_assessment',
                            'Do you think the neuroimaging was taken within temporal proximity to the lesion? Days-weeks (Good). Months-years (Bad). Unsure (Unclear).': 'lesion_related-images',
                            'Do you think another practitioner would come to the same conclusion (diagnosis/attribution) that this group did? (Good/Bad/Unclear)': 'replicability',
                            'How is the quality of this case overall? (Good/Bad/Unclear)': 'overall_appraisal'
            }        
        self.template_questions = {' ? (metric/metric/metric)': 'question_label',
                            ' ? (metric/metric/metric)': 'question_label',
                            ' ? (metric/metric/metric)': 'question_label',
                            ' ? (metric/metric/metric)': 'question_label',
                            ' ? (metric/metric/metric)': 'question_label',
                            ' ? (metric/metric/metric)': 'question_label',
                            ' ? (metric/metric/metric)': 'question_label',
                            ' ? (metric/metric/metric)': 'question_label',
            }

        
class OpenAIEvaluator:
    """
    A class to evaluate text chunks using the OpenAI API based on the type of article.

    Attributes:
    - api_key (str): OpenAI API key.
    - article_type (str): The type of article (e.g., 'research', 'case').
    - questions (dict): Dictionary mapping article types to evaluation questions.

    Methods:
    - __init__: Initializes the OpenAIEvaluator class with the API key path and article type.
    - read_api_key: Reads the OpenAI API key from a file.
    - evaluate_with_openai: Evaluates a text chunk based on the question corresponding to the article type.
    """


    def __init__(self, api_key_path):
        """
        Initializes the OpenAIEvaluator class.

        Parameters:
        - api_key_path (str): Path to the file containing the OpenAI API key.
        - article_type (str): The type of article (e.g., 'research', 'case').
        """
        self.api_key = self.read_api_key(api_key_path)
        openai.api_key = self.api_key

    def read_api_key(self, file_path):
        """
        Reads the OpenAI API key from a file.

        Parameters:
        - file_path (str): Path to the file containing the OpenAI API key.

        Returns:
        - str: OpenAI API key.
        """
        with open(file_path, 'r') as file:
            return file.readline().strip()

    def evaluate_with_openai(self, chunk, questions):
        """
        Evaluates a chunk based on multiple posed questions using OpenAI API.

        Parameters:
        - chunk (str): The text chunk to be evaluated.
        - questions (list): A list of questions for evaluation.

        Returns:
        - dict: A dictionary where keys are questions and values are binary decisions (0 or 1).
        """
        question_list = list(questions.keys())
        question_prompt = "\n".join([f"{q}" for q in question_list])
        prompt = f"Text Chunk: {chunk}\n{question_prompt}"

        try:
            response = openai.Completion.create(
                engine="gpt-3.5-turbo-16k-0613",
                prompt=prompt,
                max_tokens=10  # Adjust as needed
            )
            decision_text = response.choices[0].text.strip()
            decisions = decision_text.split("\n")
            
            if len(decisions) != len(questions):
                print("Warning: The number of decisions does not match the number of questions.")
                valid_decisions = [line.strip() for line in decisions if line.strip()]
                if len(valid_decisions) != len(questions):
                    decisions = valid_decisions
                    print("Solved warning.")
                else:
                    print('Decisions here, returning None: ', decisions)
                    return None

            return {q: 1 if "Y" in d else 0 for q, d in zip(questions, decisions)}
            
        except Exception as e:
            print(f"An error occurred: {e}")
            return "Unidentified"

class OpenAIChatEvaluator(OpenAIEvaluator):
    """
    Class to evaluate text chunks using OpenAI's chat models.
    
    Attributes:
    - token_limit (int): The maximum number of tokens allowed in each OpenAI API call.
    - question_token (int): The number of tokens reserved for the question.
    - answer_token (int): The number of tokens reserved for the answer.
    - json_data (dict): The data read from the JSON file.
    - keys_to_consider (list): List of keys to consider from the JSON file.
    - article_type (str): The type of article (e.g., 'research', 'case').
    - questions (dict): Dictionary mapping article types to evaluation questions.
    """
    
    def __init__(self, api_key_path, json_file_path, keys_to_consider, question_type, question, token_limit=16000, question_token=500, answer_token=500, debug=False):
        """
        Initializes the OpenAIChatEvaluator class.
        
        Parameters:
        - api_key_path (str): Path to the file containing the OpenAI API key.
        - json_file_path (str): Path to the JSON file containing the text data.
        - keys_to_consider (list): List of keys to consider from the JSON file.
        - article_type (str): The type of article (e.g., 'research', 'case').
        - token_limit (int): The maximum number of tokens allowed in each OpenAI API call. Default is 16000.
        - question_token (int): The number of tokens reserved for the question. Default is 500.
        - answer_token (int): The number of tokens reserved for the answer. Default is 500.
        """
        super().__init__(api_key_path)  # Call the parent class's constructor
        self.questions = question
        self.token_limit = token_limit
        self.question_token = question_token
        self.answer_token = answer_token
        self.json_path = json_file_path
        self.json_data = self.read_json(json_file_path)
        self.keys_to_consider = keys_to_consider
        self.question_type = question_type
        self.extract_relevant_text()
        self.all_answers = {}
        self.debug = debug

    def read_json(self, json_file_path):
        """
        Reads JSON data from a file.
        
        Parameters:
        - json_file_path (str): Path to the JSON file containing the text data.
        
        Returns:
        - dict: The data read from the JSON file.
        """
        try:
            with open(json_file_path, 'r') as file:
                return json.load(file)
        except FileNotFoundError:
            print(f"Error: File {json_file_path} not found.")
            return {}
        except json.JSONDecodeError:
            print("Error: Could not decode the JSON file.")
            return {}

    
    def extract_relevant_text(self):
        """
        Extracts and stores relevant text sections based on keys_to_consider.
        """
        self.relevant_text_by_file = {}
        for file_name, sections in self.json_data.items():
            selected_text = ""
            for key, value in sections.items():
                if key in self.keys_to_consider:
                    selected_text += value
            self.relevant_text_by_file[file_name] = selected_text

    def evaluate_all_files(self):
        for file_name, selected_text in tqdm(self.relevant_text_by_file.items()):
            # Initialize a dictionary to store answers for this file
            self.all_answers[file_name] = {}
            if self.debug:
                print('On file:', file_name)
            
            # Chunk the text
            text_chunker = TextChunker(selected_text, np.round((self.token_limit) * 0.7))
            text_chunker.chunk_text()
            chunks = text_chunker.get_chunks()
            if self.debug:
                print('Number of chunks:', len(chunks))
            
            # Initialize a dictionary to store chunk-level answers for each question
            for question in self.questions.keys():
                self.all_answers[file_name][question] = {}

            # Send a query for each chunk
            for chunk_index, chunk in enumerate(chunks):
                # Reset the conversation each time
                conversation = []
                conversation.append({"role": "system", "content": "You are a helpful assistant."})
                conversation.append({"role": "user", "content": f"Text Chunk: {chunk}"})
                if self.debug:
                    print('On chunk:', chunk_index)
                
                # Initialize a conversation with OpenAI for this chunk
                for q_index, q in enumerate(self.questions.keys()):
                    # Use a while loop to allow 3 submission attempts
                    retry_count = 0
                    while retry_count < 3:
                        try:
                            # Add the question to the conversation and send it
                            conversation.append({"role": "user", "content": q})
                            response = openai.ChatCompletion.create(
                                model="gpt-3.5-turbo-16k",
                                messages=conversation
                            )
                            
                            # Retrieve the assistant's last answer
                            answer = response['choices'][-1]['message']['content']
                            
                            # Store the answer for this question and this chunk
                            self.all_answers[file_name][q][f"chunk_{chunk_index+1}"] = answer
                            
                            # Add the assistant's answer back to the conversation to maintain context
                            conversation.append({"role": "assistant", "content": answer})
                            
                            time.sleep(0.1)
                            break  # Exit the loop if successful
                        
                        #Handle Exceptions
                        except Exception as e:
                            if type(e).__name__ == 'RateLimitError':
                                print(f"Rate limit error: {e}. Retrying... ({retry_count+1})")
                                retry_count += 1
                                time.sleep(30)
                            else:
                                print(f"An error occurred: {e}. Retrying... ({retry_count+1})")
                                retry_count += 1
                                time.sleep(5)
                                            
                    if retry_count == 3:
                        self.all_answers[file_name][q][f"chunk_{chunk_index+1}"] = "Unidentified"
                    
        return self.all_answers
        
    def send_to_openai(self, chunks):
        """
        Sends text chunks to OpenAI for evaluation.
        
        Parameters:
        - chunks (list): List of text chunks to evaluate.
        
        Returns:
        - list: List of answers received from OpenAI.
        """
        answers = []
        for chunk in chunks:
            prompt = f"Text Chunk: {chunk}\n{self.questions}"

            try:
                response = openai.Completion.create(
                    engine="gpt-3.5-turbo-16k",
                    prompt=prompt,
                    max_tokens=self.answer_token  # Adjust as needed
                )
                decision_text = response.choices[0].text.strip()
                answers.append(decision_text)

            except Exception as e:
                print(f"An error occurred during response handling: {e}")
                answers.append("Unidentified")

        return answers
    
    def save_to_json(self, output_dict):
        """
        Saves the labeled sections to a JSON file.

        Parameters:
        - output_dict (dict): Dictionary containing the labeled sections.

        Returns:
        - None
        """
        # Create a new directory in the same root folder
        out_dir = os.path.join(os.path.dirname(self.json_path), "text_evaluations")
        os.makedirs(out_dir, exist_ok=True)
        
        # Save the dictionary to a JSON file
        with open(os.path.join(out_dir, f'{self.question_type}_evaluations.json'), 'w') as f:
            json.dump(output_dict, f, indent=4)