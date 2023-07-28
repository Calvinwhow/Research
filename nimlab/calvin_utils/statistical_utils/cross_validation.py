
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from calvin_utils.statistical_utils.custom_regressions import generalizable_multivariate_regression
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr

def k_folds(data_df, train_function, k=10):
    #Initialize outcomes
    
    kf = KFold(n_splits=k, random_state=None)
    predictions = []
    mse = []
    pearson_r = []
    pearson_p = []
    for train_index , test_index in kf.split(data_df.iloc[:, 1:]):
        train_df = data_df.loc[train_index]
        test_df = data_df.loc[test_index, :]
        trained_results, formula = eval(train_function) #Will run as 'train_df' as the df. 
        predictions.append(trained_results.predict(test_df.iloc[:, 1:]))
        mse.append(mean_squared_error(trained_results.predict(test_df.iloc[:, 1:]), test_df.iloc[:, 0]))
        try:
            r, p = pearsonr(trained_results.predict(test_df.iloc[:, 1:]), test_df.iloc[:, 0])
            pearson_r.append(r)
            pearson_p.append(p)
        except:
            pearson_r.append(0)
            pearson_p.append(0)           
    
    avg_mse = sum(mse)/k
    avg_pearson_p = sum(pearson_p)/k
    avg_pearson_r = sum(pearson_r)/k    
    results_df = pd.DataFrame({'Avg MSE': avg_mse, 'avg_pearson_p': avg_pearson_p, 'avg_pearson_r': avg_pearson_r}, index=[0])
    return results_df

def leave_one_out(data_df, outcome_var, function, outcomes=None, leave_out_variable='index'):
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
    if outcomes is not None:
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
        results, formula = eval(function);
        prediction = results.predict(data_df.loc[i, :])
        predictions.append(prediction.values[0])
        
        #Place outcomes of interest in associated lists
        if outcomes is not None:
            for outcome in outcomes:
                results_string = 'results.'+outcome
                outcome_dictionary[outcome].extend([eval(results_string)])

    outcome_dictionary['predictions'] = predictions
    outcome_dictionary['actuals'] = data_df[outcome_var].values.tolist()
    
    try:
        outcome_dictionary['loocv_prediction_r'], outcome_dictionary['prediction_p'] = pearsonr(outcome_dictionary['actuals'], outcome_dictionary['predictions'])
        outcome_dictionary['loocv_prediction_r2'] = outcome_dictionary['prediction_r']**2
        outcome_dictionary['loocv_mean_squared_error'] = np.sum( np.square( (outcome_dictionary['actuals'] - outcome_dictionary['predictions']))) / len(outcome_dictionary['predictions'])
        outcome_dictionary['loocv_root_mean_squared_error'] = np.sqrt(outcome_dictionary['loocv_mean_squared_error'])
        outcome_dictionary['lootcv_mean_absolute_error'] = np.sum(np.abs((outcome_dictionary['predictions'] - outcome_dictionary['actuals']))) / len(outcome_dictionary['predictions'])
    except:
        print('Could not calculate outcomes, suspect NaNs')
    
    outcome_df = pd.DataFrame(outcome_dictionary)
    #Display results
    return outcome_df