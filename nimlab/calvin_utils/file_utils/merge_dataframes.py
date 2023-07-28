import pandas as pd
import os

def merge_dataframe_columns(*args):
    ''' 
    This function will take a pandas dataframe's column, identified by the syntax df['<column>']
    It then merges that number of columns into a new dataframe and returns the new dataframe
    '''
    new_df = None
    for arg in args: 
        #Remove index and join the dataframes
        new = arg.reset_index(drop=True)
        if new_df is not None:
            new_df = pd.concat([new_df, new], axis=1)#, ignore_index=True)
        else:
            new_df = new
    return new_df

def merge_dataframes(*args):
    '''Merge dataframes'''
    new_df = None
    new_names = []
    for arg in args: 
        #Remove index and join the dataframes
        new = arg.reset_index(drop=True)
        new_names.append(new.columns.values.tolist())
        if new_df is not None:
            new_df = pd.concat([new_df, new], axis=1, ignore_index=True)
        else:
            new_df = new
            
    #Prepare the new names
    column_names = []
    for list in new_names:
        for values in list:
            column_names.append(values)
    print(column_names)    
    new_df.columns = column_names
    return new_df

def merge_csv_files(*args, on_column):
    # Read in the first file to initialize the merged dataframe
    df_merged = pd.read_csv(args[0])
    
    # Merge the remaining files on the specified column
    for path in args[1:]:
        df = pd.read_csv(path)
        df_merged = pd.merge(df_merged, df, on=on_column, right_index=False, how='inner')
    
    # Save the merged dataframe as csv with the combined path basenames + '_merged'
    merged_filename = '_'.join([os.path.splitext(os.path.basename(path))[0] for path in args]) + '_merged.csv'
    out_dir = os.path.join(os.path.dirname(args[0]), os.path.basename(merged_filename))
    df_merged.to_csv(out_dir, index=False)
    
    return df_merged
