from scipy.stats import zscore
import numpy as np

def z_score_matrix(matrix):
    print('--------------------------------Performing z_score--------------------------------')
    print('pre z score max: ', np.max(matrix), np.shape(matrix))
    z_matrix = zscore(matrix)
    print('post z score max: ', np.max(z_matrix), np.shape(z_matrix))
    return z_matrix
