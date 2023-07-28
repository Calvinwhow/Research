from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from scipy.stats import pearsonr
import numpy as np

#Generate regression for whatever you want, with degree being the polynomial's degree
def lin_regression(x, y, z, degree):
    #----Generate Model
    #Prep data
    x = x
    y = y
    z = z
    input_matrix = np.stack([x, y]).T

    #----Linear Regression
    poly = PolynomialFeatures(degree=degree)
    model = LinearRegression()
    in_features = poly.fit_transform(input_matrix)
    model.fit(in_features, z)
    predicted_z = model.predict(poly.transform(input_matrix))

    #----Performance Metrics
    #Pearson of Predicted Z to Actual Z
    r, p = pearsonr(predicted_z, z) #need to find the prediction points at the x/y for each z_actual
    print('r: ', r)
    print('p: ', p)
    #Coefficients of the Regression
    coefficients = dict(zip(poly.get_features_out(), model.coef_.round(4)))
    print('coeff: ', coefficients)
    #Check Fit
    r_squared = model.score(poly.transform(input_matrix), z)
    print('r2: ', r_squared)

    #----Generate Planes for Plotting
    x_lin=np.linspace(np.min(x), np.max(x), 100)
    y_lin=np.linpsace(np.min(y), np.max(y), 100)
    X_plane,Y_plane=np.meshgrid(x_lin,y_lin,copy=False)
    input_planes=np.stack([X_plane,Y_plane]).T
    assert(input_planes.shape==(100*100, 2)) #unsure what shape 400,2 refers to
    predicted_plane = model.predict(poly.transform(input_planes))


    return coefficients, r_squared, r, p, X_plane, Y_plane, predicted_plane

def basic_trivariate_regression(x, y, z, dataframe):
    import statsmodels.formula.api as smf
    import pandas as pd
    
    formula = f'{z} ~ {x} + {y}'
    model = smf.ols(formula, data=dataframe)
    results = model.fit()
    return results

def generalizable_multivariate_regression(dataframe, dep_var_string, indep_var_string_list):
    import statsmodels.formula.api as smf
    import pandas as pd    
    
    formula = ''
    #generate the string in parsable format for statsmodels
    for string in indep_var_string_list:
        if formula == '':
            formula = string
        else:
            formula = formula + ' + ' + string
    
    formula = dep_var_string + ' ~ ' + formula
    model = smf.ols(formula, data=dataframe)
    results = model.fit()
    return results, formula