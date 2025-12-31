import os
import json
import time
import copy
import heapq
from multiprocessing import Pool, cpu_count

DATA_PATH = "./data/data.txt"
SAVE_PATH = "./result/HC/HC_log_max.jsonl"
DISTANCE_MODE = "max"

class Hierarchical_Clustering:
    def __init__(self, data):
        self.q = []
        self.n = len(data)
        self.cluster = {}
        self.next_cluster =  self.n
        for i in range(self.n):
            self.cluster.update({i: [data[i]]})
        
        for i in range(self.n):
            for j in range(i + 1, self.n):
                x1, y1 = data[i]
                x2, y2 = data[j]
                d = (x1 - x2)*(x1 - x2) + (y1 - y2)*(y1 - y2)
                heapq.heappush(self.q, (d, (i, j)))
            
            percent = i / self.n * 100
            print(f"\rInitial distance: {percent:6.2f}%", end="")


    def min_linkage(self, c1: int, c2: int) -> float:
        min_dist = float("inf")
        for x1, y1 in self.cluster[c1]:
            for x2, y2 in self.cluster[c2]:
                dx = x1 - x2
                dy = y1 - y2
                d = dx*dx + dy*dy
                if d < min_dist:
                    min_dist = d
        return min_dist


    def max_linkage(self, c1: int, c2: int) -> float:
        max_dist = 0.0
        for x1, y1 in self.cluster[c1]:
            for x2, y2 in self.cluster[c2]:
                dx = x1 - x2
                dy = y1 - y2
                d = dx*dx + dy*dy
                if d > max_dist:
                    max_dist = d
        return max_dist


    def average_linkage(self, c1: int, c2: int) -> float:
        total = 0.0
        count = 0
        for x1, y1 in self.cluster[c1]:
            for x2, y2 in self.cluster[c2]:
                dx = x1 - x2
                dy = y1 - y2
                total += dx*dx + dy*dy
                count += 1
        return total / count


    def trian(self, dist_func: str):
        if dist_func == "min":
            get_dist = self.min_linkage
        elif dist_func == "max":
            get_dist = self.max_linkage
        else:
            get_dist = self.average_linkage

        # import pdb
        # pdb.set_trace()
        result = []
        while len(self.cluster) != 1:
            _, (x, y) = heapq.heappop(self.q)
            if x not in self.cluster or y not in self.cluster:
                continue
            new_cluster = {self.next_cluster: self.cluster[x] + self.cluster[y]}
            self.cluster.update(new_cluster)
            self.cluster.pop(x)
            self.cluster.pop(y)

            for clu in self.cluster:
                if clu != self.next_cluster:
                    d = get_dist(clu, self.next_cluster)
                    heapq.heappush(self.q, (d, (clu, self.next_cluster)))
            
            self.next_cluster += 1
            if len(self.cluster) <= 20:
                result.append(copy.deepcopy(self.cluster))
            percent = (5000 - len(self.cluster)) / 5000 * 100
            print(f"\rProgress: {percent:6.2f}%", end="")
            
        return result

def main():
    start = time.time()
    with open(DATA_PATH, "r") as fr:
        data = [tuple(map(float, line.split())) for line in fr.readlines()]
    
    hc = Hierarchical_Clustering(data)
    print()
    result = hc.trian(DISTANCE_MODE)
    with open(SAVE_PATH, "w") as fw:
        for res in result:
            fw.write(json.dumps(res))
            fw.write("\n")
    end = time.time()
    print()
    print(f"{end - start:.6f} seconds")

if __name__ == "__main__":
    main()