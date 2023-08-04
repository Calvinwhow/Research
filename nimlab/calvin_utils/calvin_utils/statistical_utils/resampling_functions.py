from tqdm import tqdm
import pandas as pd
import numpy as np
class Bootstrap:
    def __init__(self, data, func, func_args=None, bootstrap_samples=2500):
        self.data = data
        self.func = func
        self.func_args = func_args if func_args is not None else {}
        self.bootstrap_samples = bootstrap_samples

    def bootstrap_function(self):
        bootstrap_results = []
        for _ in tqdm(range(self.bootstrap_samples)):
            sample = self.data.sample(frac=1, replace=True)
            bootstrap_results.append(self.func(sample, **self.func_args))
        return bootstrap_results
    
    
    