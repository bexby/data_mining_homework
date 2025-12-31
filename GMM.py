import os
import time
import json
import math
import random
from typing import List, Callable, Tuple
from functools import partial
from multiprocessing import Pool, cpu_count


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




# def initialize_kmeans_builtin_clusters(data: List[List], k: int):
#     def euclidean_distance(p1, p2):
#         return ((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)**0.5

#     def covariance_weight(x, y):
#         n = len(x)
#         mean_x = sum(x)/n
#         mean_y = sum(y)/n
#         return sum((xi-mean_x)*(yi-mean_y) for xi, yi in zip(x, y)) / n
    
#     mu = [random.choice(data)]
#     while len(mu) < k:
#         dist_sq = []
#         for point in data:
#             d = min(euclidean_distance(point, center)**2 for center in mu)
#             dist_sq.append(d)
#         next_center = data[dist_sq.index(max(dist_sq))]
#         mu.append(next_center)

#     clusters = [[] for _ in range(k)]
#     for point in data:
#         distances = [euclidean_distance(point, center) for center in mu]
#         cluster_idx = distances.index(min(distances))
#         clusters[cluster_idx].append(point)

#     sigma = []
#     for cluster in clusters:
#         if len(cluster) == 0:
#             xlis = [d[0] for d in data]
#             ylis = [d[1] for d in data]
#         else:
#             xlis = [d[0] for d in cluster]
#             ylis = [d[1] for d in cluster]
#         x_var = covariance_weight(xlis, xlis)
#         y_var = covariance_weight(ylis, ylis)
#         cov_xy = covariance_weight(xlis, ylis)
#         sigma.append([[x_var, cov_xy], [cov_xy, y_var]])

#     total_points = len(data)
#     pi = [len(cluster)/total_points for cluster in clusters]

#     return pi, mu, sigma



def initialize_kmeans_builtin_clusters(data: List[List], k: int):
    def euclidean_distance(p1, p2):
        return ((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)**0.5
    # 1. KMeans++ 风格初始化质心
    mu = []
    # 随机选择第一个质心
    mu.append(random.choice(data))
    
    while len(mu) < k:
        # 计算每个点到最近已有质心的距离平方
        dist_sq = []
        for point in data:
            d = min(euclidean_distance(point, center)**2 for center in mu)
            dist_sq.append(d)
        # 选择距离平方最大的点作为下一个质心
        next_center = data[dist_sq.index(max(dist_sq))]
        mu.append(next_center)
    
    # 2. 计算全局协方差（和你原来一样）
    xlis = [d[0] for d in data]
    ylis = [d[1] for d in data]
    x_var = covariance_weight(xlis, xlis)
    y_var = covariance_weight(ylis, ylis)
    cov_xy = covariance_weight(xlis, ylis)
    sigma = [[[x_var, cov_xy], [cov_xy, y_var]] for _ in range(k)]

    # 3. 混合系数均等
    pi = [1/k for _ in range(k)]

    return pi, mu, sigma




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


    def train(self, data: List[List], steps: int, accelerate: bool = False, init_value=None):

        if init_value is not None:
            self.alpha, self.mu, self.sigma = init_value
        else:
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
    '''
        运行1个k值
    '''
    start = time.time()
    with open(DATA_PATH, "r") as fr:
        data = [list(map(float, line.split())) for line in fr.readlines()]

    gmm = Gaussian_Mixture_Model_2d(K, INITIAL_METHOD)
    # gmm = Gaussian_Mixture_Model_2d(K, initialize_kmeans_builtin_clusters)
    log = gmm.train(data, STEPS, ACCELERATE)
    
    with open(SAVE_PATH, "w") as fw:
        for l in log:
            fw.write(json.dumps(l))
            fw.write("\n")
            
    end = time.time()
    print(f"{end - start:.6f} seconds")


def main2(min_k=10, max_k=20, results_path="bic_aic_results.json", accelerate=True):
    '''
        运行多个k值
    '''

    def num_gmm_params(k: int, d: int) -> int:
        return (k - 1) + k * d + k * d * (d + 1) // 2

    with open(DATA_PATH, "r") as fr:
        data = [list(map(float, line.split())) for line in fr.readlines()]

    n_samples = len(data)
    d = len(data[0])
    results = []

    start_all = time.time()
    for k in range(min_k, max_k + 1):
        t0 = time.time()
        print(f"Training GMM with K={k} ...")
        gmm = Gaussian_Mixture_Model_2d(k, custom_initialize)
        log = gmm.train(data, STEPS, accelerate=accelerate)

        last_entry = log[-1]
        loss = last_entry.get("loss", None)
        if loss is None:
            loss = 0.0
            for dpt in data:
                probs = map(lambda mu_sigma: None, [])  
        log_likelihood = -loss  

        p = num_gmm_params(k, d)
        bic = p * math.log(n_samples) - 2.0 * log_likelihood
        aic = 2.0 * p - 2.0 * log_likelihood

        results.append({
            "K": k,
            "n_samples": n_samples,
            "dim": d,
            "num_params": p,
            "log_likelihood": log_likelihood,
            "BIC": bic,
            "AIC": aic,
            "train_time_seconds": time.time() - t0
        })

        with open(results_path, "w") as fw:
            json.dump(results, fw, indent=2, ensure_ascii=False)

        print(f" K={k} done, logL={log_likelihood:.3f}, params={p}, BIC={bic:.3f}, AIC={aic:.3f}")

    print(f"All done in {time.time() - start_all:.2f} s. Results saved to {results_path}")


K = 5
STEPS = 100
ACCELERATE = True
DATA_PATH = "./data/data.txt"
SAVE_PATH = f"./result/GMM/GMM_log_{K}_kmeans.jsonl"
INITIAL_METHOD = custom_initialize


if __name__ == "__main__":
    main()
    # main2()