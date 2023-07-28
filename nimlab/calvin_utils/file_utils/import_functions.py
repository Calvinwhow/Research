## Paths Input Here
from matplotlib import pyplot as plt
import warnings
warnings.filterwarnings('ignore')
import os 
def paths_to_input_files(path_1=None, path_2=None, analysis_name=None):
    analysis = "default_analysis"

    out_dir = os.path.join(os.path.dirname(path_1), f'{analysis}')
    if os.path.isdir(out_dir) != True:
        os.makedirs(out_dir)
    print('I will save to:', out_dir)
    return path_1, path_2, out_dir