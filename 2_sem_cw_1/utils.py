import numpy as np
from copy import deepcopy


def norm_1_vec(x):
    x = np.array(deepcopy(x))
    return np.max(np.abs(x))


def norm_2_vec(x):
    x = np.array(deepcopy(x))
    return np.sum(np.abs(x))


def norm_3_vec(x):
    x = np.array(deepcopy(x))
    return np.sqrt(np.sum(x**2))


def norm_1(A):
    A = np.array(deepcopy(A))
    return np.max(np.sum(np.abs(A), axis=1))


def norm_2(A):
    A = np.array(deepcopy(A))
    return np.max(np.sum(np.abs(A), axis=0))


def norm_3(A):
    A = np.array(deepcopy(A))
    return np.sqrt(norm_spectral(A.T @ A))


def norm_spectral(A):
    A = np.array(deepcopy(A))
    eig_values = np.linalg.eigvals(A)
    eig_values_norms = np.array([np.linalg.norm(elem) for elem in eig_values])
    return np.max(eig_values_norms)


def count_norms(A):
    norm_func_collection = [norm_1, norm_2, norm_3]
    norm_values = [norm_func_collection[i](A) for i in range(len(norm_func_collection))]
    for i, value in enumerate(norm_values):
        print(f"{i+1}-ая норма: {value}")
    return np.array(norm_values)
