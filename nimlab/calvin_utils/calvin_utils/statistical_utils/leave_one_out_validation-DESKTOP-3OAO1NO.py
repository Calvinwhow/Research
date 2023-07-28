
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from calvin_utils.statistical_utils.custom_regressions import generalizable_multivariate_regression
import pandas as pd

def leave_one_out(data_df, outcome_var, function, outcomes, leave_out_variable='index'):
    '''
    This is a function which will take a dataframe, take the outcome variable's name, and take a written function. 
    It will perform a leave-one-out cross-validation.
    It will create a list which contains the names of all variables to be iterated over after, and then it will evaluate the response of the function x times.
    '''
    
    #Create a list of the variables to iterate over
    if leave_out_variable == 'index':
        leave_out = data_df.index.values.tolist()
    else:
        names = data_df.copy()
        names.pop(outcome_var)
        leave_out = names.columns.tolist()
    
    #Create a loop which performs the function
        #This was built to run generalizable_multivariate_regression
        
    #Initialize lists to hold performance outcomes
    outcome_dictionary = {}
    for outcome in outcomes:
        outcome_dictionary[outcome] = []
        outcome_dictionary[f'mean_{outcome}'] = []
        print(outcome_dictionary[outcome])
    
    predictions = []
    for i in range(0, len(leave_out)):
        #Remove the 'leave-one-out' variable
        independent_variables = leave_out.copy()
        independent_variables.pop(i)
        
        #Index the series to remain
        iterating_df = data_df.copy()
        if leave_out_variable=='index':
            iterating_df = iterating_df.loc[independent_variables, :]
        else:
            iterating_df = iterating_df.loc[:, independent_variables]
            
        #Extract results from the statsmodels results class
        results = eval(function);
        prediction = results.predict(data_df.loc[i, :])
        predictions.append(prediction.values[0])
        
        #Place outcomes of interest in associated lists
        for outcome in outcomes:
            results_string = 'results.'+outcome
            outcome_dictionary[outcome].extend([eval(results_string)])

    #Calculate average performance across trials
    for outcome in outcomes:
        outcome_dictionary[f'mean_{outcome}'] = np.mean(outcome_dictionary[outcome])
    outcome_dictionary['predictions'] = predictions
    outcome_dictionary['actuals'] = data_df[outcome_var].values.tolist()
    
    try:
        outcome_dictionary['prediction_r'], outcome_dictionary['prediction_p'] = pearsonr(outcome_dictionary['actuals'], outcome_dictionary['predictions'])
        outcome_dictionary['prediction_r2'] = outcome_dictionary['prediction_r']**2
    except:
        print('Could not calculate pearson R and P, suspect NaNs')
    
    outcome_df = pd.DataFrame(outcome_dictionary)
    #Display results
    return outcome_df