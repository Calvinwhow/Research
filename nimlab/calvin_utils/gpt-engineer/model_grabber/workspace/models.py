class LinearRegressionModel:
    def __init__(self):
        self.name = "linear_regression_model"

    def start_function(self, x, y, dataframe):
        return f"{self.name}(x={x}, y={y}, dataframe={dataframe})"


class MediationAnalysisModel:
    def __init__(self):
        self.name = "mediation_analysis_model"

    def start_function(self, x, y, z, dataframe):
        return f"{self.name}(x={x}, y={y}, z={z}, dataframe={dataframe})"
