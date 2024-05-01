import statsmodels.formula.api as smf

class RegressOutCovariates():
    """
    Will regress on values and return residuals. Will add the residuals to a dataframe as <name>_residual and return the DF
    """
    @staticmethod
    def generate_formula(independent_variable_list, covariates_list, verbose=False):
        formula_dict = {}
        for dep_var in independent_variable_list:
            formula = dep_var
            for covariate in covariates_list:
                if covariate == covariates_list[0]:
                    formula += f" ~ {covariate}"
                else:
                    formula += f" + {covariate}"
            formula_dict[dep_var] = formula
            print(f"Formula for {dep_var}: \n", formula) if verbose else None
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
    def run(df, independent_variable_list, covariates_list):
        """
        Params:
        
        df: pandas DF containing your covariates and independent variables
        independent_variable_list: a list of indendent variables as found in the dataframe columns. 
        covariates_list: a list of covariates as found in the dataframe columns. 
        """
        formula_dict = RegressOutCovariates.generate_formula(independent_variable_list, covariates_list)
        df, adjusted_indep_vars_list = RegressOutCovariates.regress_out_covariates(df, formula_dict)
        

        return df, adjusted_indep_vars_list