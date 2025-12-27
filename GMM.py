import os
import time
import json
import math
import random
from typing import List, Callable, Tuple
from functools import partial
from multiprocessing import Pool, cpu_count

K = 15
STEPS = 100
DATA_PATH = "data.txt"
SAVE_PATH = f"GMM_log_{K}.jsonl"



def matrix_mul(mat1: List[List], mat2: List[List]) ->List[List]:
    assert len(mat1[0]) == len(mat2), "mat1's columns must be equal to mat2's rows"
    result = []
    for r in mat1:
        res_row = []
        for l in zip(*mat2):
            temp = [i * j for i, j in zip(r, l)]
            res_row.append(sum(temp))
        result.append(res_row)
    
    return result


def matrix_inverse_det_2d(matrix: List[List]) -> Tuple[List[List], float]:
    assert len(matrix) == 2, "matrix must be 2 * 2 !"
    assert len(matrix[0]) == 2, "matrix must be 2 * 2 !"
    assert len(matrix[1]) == 2, "matrix must be 2 * 2 !"

    (a, b), (c, d) = matrix[0], matrix[1]
    det = a * d - b * c
    assert det != 0, "det cannot be zero"

    inv_mat = [[d, -b], [-c, a]]
    return [[item / det for item in inv_mat[j]] for j in range(2)], det


def Gaussian_Distibution_2d(data: List, mu: List, sigma: List) -> float:
    assert len(data) == 2, "data must be 2 dim"
    assert len(mu) == 2, "mu must be K dim"

    mat_inv, det = matrix_inverse_det_2d(sigma)
    c = 1 / (2*math.pi * math.sqrt(det))
    e1 = matrix_mul([[d - u for d, u in zip(data, mu)]], mat_inv)
    e2 = matrix_mul(e1, [[d - u] for d, u in zip(data, mu)])[0][0]

    return c * math.exp(-0.5 * e2)
    

def covariance_weight(x: List, y: List, weight: List = None) -> List:
    assert len(x) == len(y) if weight is None else len(x) == len(y) == len(weight), "length not equal"
    
    if weight is not None:
        W = sum(weight)
        if W == 0:
            return 0.0
        x_mean = sum(xi * wi for xi, wi in zip(x, weight)) / W
        y_mean = sum(yi * wi for yi, wi in zip(y, weight)) / W
        return sum((xi - x_mean) * (yi - y_mean) * wi for xi, yi, wi in zip(x, y, weight)) / W
    else:
        x_mean = sum(x) / len(x)
        y_mean = sum(y) / len(y)
        return sum((xi - x_mean) * (yi - y_mean) for xi, yi in zip(x, y)) / len(x)
    

def custom_initialize(data: List[List], k: int):
    xlis = [d[0] for d in data]
    ylis = [d[1] for d in data]

    mu = random.sample(data, k)
    x_var = covariance_weight(xlis, xlis)
    y_var = covariance_weight(ylis, ylis)
    cov_xy = covariance_weight(xlis, ylis)
    sigma = [[[x_var, cov_xy], [cov_xy, y_var]] for _ in range(k)]

    return [1/k for _ in range(k)], mu, sigma


class Gaussian_Mixture_Model_2d:
    def __init__(self, K: int, init_func: Callable):
        self.k = K
        self.init_func = init_func
        self.alpha, self.mu, self.sigma = None, None, None


    def loss(self, data):
        probs = map(Gaussian_Distibution_2d, [data] * self.k, self.mu, self.sigma)
        weighted_sum = sum(a * p for a, p in zip(self.alpha, probs))
        loss = -math.log(weighted_sum)
        return loss


    def train(self, data: List[List], steps: int, accelerate: bool = False):

        self.alpha, self.mu, self.sigma = self.init_func(data, self.k)
        log = []
        with Pool(cpu_count()) as executor:
            for step in range(steps):
                p_kn = []   # (K, N)
                for k in range(self.k):
                    Gauss = partial(Gaussian_Distibution_2d, mu=self.mu[k], sigma=self.sigma[k])
                    if accelerate:
                        p_kn.append(list(executor.map(Gauss, data)))
                    else:  
                        p_kn.append(list(map(Gauss, data)))
                
                sum_n = [sum(i * a for i, a in zip(item, self.alpha)) for item in zip(*p_kn)]  # (N, )
                gama_kn = [[a * i / s for i, s in zip(j, sum_n)] for j, a in zip(p_kn, self.alpha)] # (K, N)
                sum_gama = [sum(item) for item in gama_kn]  # (K, )
                self.alpha = [item / sum(sum_gama) for item in sum_gama] # (K, )


                new_mu = []
                new_sigma = []
                for gama, sg in zip(gama_kn, sum_gama):
                    temp_mu = [[i*ga, j*ga] for (i, j), ga in zip(data, gama)]   # (N, 2)
                    temp_mu = [sum(i) / sg for i in zip(*temp_mu)]  # (2, )
                    new_mu.append(temp_mu)

                    bias_x, bias_y = [[i - m for i in d] for d, m in zip(zip(*data), temp_mu)]
                    x_var = covariance_weight(bias_x, bias_x, gama)
                    y_var = covariance_weight(bias_y, bias_y, gama)
                    cov_xy = covariance_weight(bias_x, bias_y, gama)
                    new_sigma.append([[x_var, cov_xy], [cov_xy, y_var]])

                self.mu = new_mu
                self.sigma = new_sigma

                if accelerate:
                     loss_list = executor.map(self.loss, data)
                     loss = sum(loss_list)
                else:
                    loss = 0.0
                    for d in data:
                        probs = map(Gaussian_Distibution_2d, [d] * self.k, self.mu, self.sigma)
                        weighted_sum = sum(a * p for a, p in zip(self.alpha, probs))
                        loss += -math.log(weighted_sum)
                
                log.append({"step": step, "alpha": self.alpha, "mu": self.mu, "sigma": self.sigma, "loss": loss})
        return log


def main():
    start = time.time()
    with open(DATA_PATH, "r") as fr:
        data = [list(map(float, line.split())) for line in fr.readlines()]
    # import pdb
    # pdb.set_trace()
    gmm = Gaussian_Mixture_Model_2d(K, custom_initialize)
    log = gmm.train(data, STEPS, True)
    
    with open(SAVE_PATH, "w") as fw:
        for l in log:
            fw.write(json.dumps(l))
            fw.write("\n")
            
    end = time.time()
    print(f"{end - start:.6f} seconds")

if __name__ == "__main__":
    main()