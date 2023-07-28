# utils/cli_resource_printer.py

def print_text_file(file_path):
    '''
    This is used to present the CLI's pseudo-GUI
    
    Args: file_path - path to the txt file with the pseudo-gui contents
    
    Returns: no returns
    '''
    with open(file_path, 'r') as file:
        contents = file.read()
        print(contents)
