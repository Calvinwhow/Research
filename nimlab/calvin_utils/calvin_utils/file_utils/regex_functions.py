import re
def extract_digits(text):
    return re.findall(r'\d+', text)

def extract_xyz_coordinates_from_string(string):
    # Define a regular expression pattern to match the coordinates
    pattern = r"_(\-?\d+\.\d+)_(\-?\d+\.\d+)_(\-?\d+\.\d+)_"

    # Use the findall() method to extract all matches of the pattern
    matches = re.findall(pattern, string)

    # Create a list to hold the coordinates
    coordinates = []

    # Loop through each match and append the coordinates to the list
    for match in matches:
        x = float(match[0])
        y = float(match[1])
        z = float(match[2])
        coordinates.append((x, y, z))

    # Return the list of coordinates
    return coordinates

import pandas as pd
from typing import Tuple, List

def extract_xyz_coordinates(data_df: pd.DataFrame, column_with_coordinates: str) -> pd.DataFrame:
    """
    Extract the x, y, and z coordinates from a dataframe column containing file paths with coordinates.

    Parameters:
        data_df (pd.DataFrame): The dataframe containing the file paths.
        column_with_coordinates (str): The name of the column containing the file paths.

    Returns:
        pd.DataFrame: A new dataframe containing the x, y, and z coordinates.
    """
    xyz_df = pd.DataFrame()

    # Loop through each row in the dataframe
    for i in range(0, len(data_df.index)):
        # Extract the coordinates from the file path
        extracted_coordinates = extract_xyz_coordinates_from_string(data_df.loc[i, column_with_coordinates])

        # Add the coordinates to the new dataframe
        xyz_df.loc[i, 'x'] = extracted_coordinates[0][0]
        xyz_df.loc[i, 'y'] = extracted_coordinates[0][1]
        xyz_df.loc[i, 'z'] = extracted_coordinates[0][2]

    # Return the new dataframe
    return xyz_df

def extract_xyz_coordinates_from_string(string: str) -> List[Tuple[float, float, float]]:
    """
    Extract the x, y, and z coordinates from a string containing a file path.

    Parameters:
        string (str): The file path containing the coordinates.

    Returns:
        List[Tuple[float, float, float]]: A list of tuples containing the x, y, and z coordinates.
    """
    # Define a regular expression pattern to match the coordinates
    pattern = r"_(\-?\d+\.\d+)_(\-?\d+\.\d+)_(\-?\d+\.\d+)_"

    # Use the findall() method to extract all matches of the pattern
    matches = re.findall(pattern, string)

    # Create a list to hold the coordinates
    coordinates = []

    # Loop through each match and append the coordinates to the list
    for match in matches:
        x = float(match[0])
        y = float(match[1])
        z = float(match[2])
        coordinates.append((x, y, z))

    # Return the list of coordinates
    return coordinates
