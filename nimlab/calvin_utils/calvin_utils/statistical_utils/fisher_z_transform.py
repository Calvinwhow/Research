import numpy as np
def fisher_z_transform(matrix):
    ##Perform fisher z transformation on r matrix. 
    rfz_matrix = np.arctanh(matrix)
    return rfz_matrix