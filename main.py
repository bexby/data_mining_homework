import os
import math
from typing import List, Optional
from functools import partial
from concurrent.futures import ProcessPoolExecutor

DATA_PATH = ""
K = 3

def matrix_inverse_2d(matrix: List[List]) -> List[List]:
    assert len(matrix) == 2, "matrix must be 2 * 2 !"
    assert len(matrix[0]) == 2, "matrix must be 2 * 2 !"
    assert len(matrix[1]) == 2, "matrix must be 2 * 2 !"

    (a, b), (c, d) = matrix[0], matrix[1]
    det = a * d - b * c
    assert det != 0, "det cannot be zero"
    inv_mat = [[d, -b], [-c, a]]
    return [[item / det for item in inv_mat[j]] for j in range(2)]


def Gaussian_Distibution(data: List, mu: List, sigma: List) -> float:
    assert len(data) == 2, "data must be 2 dim"
    assert len(mu) == K, "mu must be K dim"


def train():
    pass


def main():
    pass


if __name__ == "__main__":
    main()