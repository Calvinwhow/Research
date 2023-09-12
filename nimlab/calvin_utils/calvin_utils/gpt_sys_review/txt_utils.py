import os
import re
import json

class TextChunker:
    """
    A class to chunk a given text into smaller segments based on a token limit.
    
    Attributes:
    - text (str): The text to be chunked.
    - token_limit (int): The maximum number of tokens allowed in each chunk.
    - chunks (list): List to store the generated text chunks.
    
    Methods:
    - chunk_text: Splits the text into smaller segments based on the token limit.
    - get_chunks: Returns the list of generated text chunks.
    
    Example Usage:
    # # Setting the token limit to 75% of GPT-3's maximum token limit (4096)
    # token_limit = int(0.75 * 4096)  # About 3072 tokens

    # # Reading the text file
    # file_path = '/mnt/data/Horn 2017PD Fxconn_OCR.txt'
    # with open(file_path, 'r', encoding='utf-8') as file:
    #     text = file.read()

    # # Creating an instance of the TextChunker class
    # text_chunker = TextChunker(text, token_limit)

    # # Chunking the text
    # text_chunker.chunk_text()

    # # Getting the list of chunks
    # chunks = text_chunker.get_chunks()

    # # Displaying the first chunk as a sample
    # chunks[0][:500]  # Displaying the first 500 characters of the first chunk as a sample
    """
    
    def __init__(self, text, token_limit):
        """
        Initializes the TextChunker class with the text and token limit.
        
        Parameters:
        - text (str): The text to be chunked.
        - token_limit (int): The maximum number of tokens allowed in each chunk.
        """
        self.text = text
        self.token_limit = token_limit
        self.chunks = []
    
    def chunk_text(self):
        """
        Splits the text into smaller segments based on the token limit.
        """
        words = self.text.split()
        current_chunk = []
        current_chunk_tokens = 0
        
        for word in words:
            # Considering each word as a token and adding 1 for the space
            tokens_in_word = len(word.split()) + 1
            
            if current_chunk_tokens + tokens_in_word <= self.token_limit:
                current_chunk.append(word)
                current_chunk_tokens += tokens_in_word
            else:
                self.chunks.append(' '.join(current_chunk))
                current_chunk = [word]
                current_chunk_tokens = tokens_in_word
        
        # Adding the last chunk if any words are left
        if current_chunk:
            self.chunks.append(' '.join(current_chunk))
    
    def get_chunks(self):
        """
        Returns the list of generated text chunks.
        
        Returns:
        - list: List containing the generated text chunks.
        """
        return self.chunks

            
