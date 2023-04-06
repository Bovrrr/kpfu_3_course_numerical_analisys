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
    return np.sqrt(norm_spectral(A @ A.T))


def norm_spectral(A):
    A = np.array(deepcopy(A))
    eig_values = np.linalg.eigvals(A)
    eig_values_norms = np.array([np.linalg.norm(elem) for elem in eig_values])
    return np.max(eig_values_norms)


def count_norms(A):
    norm_func_collection = [norm_1, norm_2, norm_3]
    norm_values = [norm_func_collection[i](A) for i in range(len(norm_func_collection))]
    # for i, value in enumerate(norm_values):
    #     print(f"{i+1}-ая норма: {value}")
    return np.array(norm_values)


def make_B_b(A, y):
    B = -np.copy(A)
    for i in range(B.shape[0]):
        B[i, i] += 1
    b = np.copy(y)
    return B, b


def make_F_H(B):
    F = np.triu(B)
    H = B - F
    return F, H


def make_B__b_(F, H, b):
    n = F.shape[0]
    B_ = np.linalg.inv(np.eye(n) - H) @ F
    b_ = np.linalg.inv(np.eye(n) - H) @ b
    return B_, b_


def check_norm(A, y, D):
    A, y = D @ A, D @ y
    B, b = make_B_b(A, y)
    F, H = make_F_H(B)
    B_, b_ = make_B__b_(F, H, b)
    q_col = count_norms(B_)
    return [q_col.min(), q_col.max()]