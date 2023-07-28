import os

class FileWriter:
    @staticmethod
    def write_to_file(filename: str, text: str, append: bool) -> None:
        mode = 'a' if append else 'w'
        with open(filename, mode) as file:
            file.write(text)

