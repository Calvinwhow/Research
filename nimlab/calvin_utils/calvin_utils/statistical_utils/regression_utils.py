import statsmodels.formula.api as smf

class RegressOutCovariates():
    """
    Will regress on values and return residuals. Will add the residuals to a dataframe as <name>_residual and return the DF
    """
    @staticmethod
    def generate_formula(dependent_variable_list, covariates_list, intercept, verbose=True):
        formula_dict = {}
        for dep_var in dependent_variable_list:
            formula = dep_var
            for covariate in covariates_list:
                if covariate == covariates_list[0]:
                    formula += f" ~ {covariate}"
                else:
                    formula += f" + {covariate}"
            if intercept:
                pass
            else:
                 formula += f" - 1"
            formula_dict[dep_var] = formula
            print(f"\n Formula for {dep_var}: \n", formula) if verbose else None
        return formula_dict
    
    @staticmethod
    def regress_out_covariates(df, formula_dict):
        adjusted_indep_vars_list = []
        for indep_var, formula in formula_dict.items():
            fitted_model = smf.ols(data=df, formula=formula).fit()
            residuals = fitted_model.resid
            df[f"{indep_var}_residual"] = residuals
            adjusted_indep_vars_list.append(f"{indep_var}_residual")
        return df, adjusted_indep_vars_list
    
    @staticmethod
    def run(df, dependent_variable_list, covariates_list, intercept=True):
        """
        Params:
        
        df: pandas DF containing your covariates and independent variables
        dependent_variable_list: a list of dependent variables as found in the dataframe columns. 
        covariates_list: a list of covariates as found in the dataframe columns. 
        """
        formula_dict = RegressOutCovariates.generate_formula(dependent_variable_list, covariates_list, intercept)
        df, adjusted_indep_vars_list = RegressOutCovariates.regress_out_covariates(df, formula_dict)
        return df, adjusted_indep_vars_list