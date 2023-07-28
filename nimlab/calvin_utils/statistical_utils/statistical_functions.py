import pandas as pd
import itertools

def generate_interaction_features(df):
    """Generate interaction features between every column in a dataframe and appends them to a new column"""
    new_df = pd.DataFrame()
    columns = df.columns
    
    for c1, c2 in itertools.combinations(columns, 2):
        feature_name = f"{c1}_x_{c2}"
        new_df[feature_name] = df[c1] * df[c2]
        
    return pd.concat([df, new_df], axis=1)
