import os
import time
import heapq as hq
import numpy as np
import pandas as pd
from tqdm import tqdm
import pdb

class Collaborative_Filtering:
    def __init__(self, save_topk: int):
        self.movie_similary = dict()
        self.save_topk = save_topk
        self.data = None
        self.group_dict = dict()    # 每部电影被哪些用户看过
        self.score_dict = dict()    # 每部电影被哪些用户看过相应的评分
        self.user_dict = dict()     # 每个用户看过哪些电影


    def cosin_similary(self, m1, m2):
        common_user = set(self.group_dict[m1]).intersection(self.group_dict[m2])
        if len(common_user) == 0:
            return None

        rating_m1 = []
        for idx, user in enumerate(self.group_dict[m1]):
            if user in common_user:
                rating_m1.append(self.score_dict[m1][idx])

        rating_m2 = []
        for idx, user in enumerate(self.group_dict[m2]):
            if user in common_user:
                rating_m2.append(self.score_dict[m2][idx])
        
        arr_m1, arr_m2 = np.array(rating_m1), np.array(rating_m2)
        return np.dot(arr_m1, arr_m2) / (np.linalg.norm(arr_m1) * np.linalg.norm(arr_m2))

    def keep_topk_sim(self, heap_m, insert_m, sim):
        small_heap = self.movie_similary[heap_m]
        if len(small_heap) < self.save_topk:
            hq.heappush(small_heap, (sim, insert_m))
        else:
            if sim > small_heap[0][0]:
                hq.heapreplace(small_heap, (sim, insert_m))

    def train(self, data: pd.DataFrame):
        movie = list(set(data["movieId"]))
        for m in movie:
            self.movie_similary.update({m: []})

        for row in data.itertuples(index=False):
            if row.userId not in self.user_dict:
                self.user_dict.update({row.userId: [row.movieId]})
            else:
                self.user_dict[row.userId].append(row.movieId)
            
            if row.movieId not in self.group_dict:
                self.group_dict.update({row.movieId: [row.userId]})
                self.score_dict.update({row.movieId: [row.rating]})
            else:
                self.group_dict[row.movieId].append(row.userId)
                self.score_dict[row.movieId].append(row.rating)

        for i1, m1 in enumerate(movie):
            for i2 in range(i1 + 1, len(movie)):
                m2 = movie[i2]

                sim = self.cosin_similary(m1, m2)
                if sim is not None:
                    if m1 == 193609 or m2 == 193609:
                        pdb.set_trace()
                    self.keep_topk_sim(m1, m2, sim)
                    self.keep_topk_sim(m2, m2, sim)
            
            percent = i1 / len(movie) * 100
            print(f"\rProgress: {percent:6.2f}%", end="")



    def prediection(self, user_id, movie_id):
        user_seen = self.user_dict[user_id]
        candidate = []
        for s, movie in self.movie_similary[movie_id]:
            if movie in user_seen:
                for idx, u in enumerate(self.group_dict[movie]):
                    if u == user_id:
                        candidate.append((movie, s, self.score_dict[idx]))
                        break
        
        pred = sum(item[1] * item[2] for item in candidate) / sum(item[1] for item in candidate)
        return pred

def main():
    df = pd.read_csv("./data/ml-latest-small/ratings.csv")
    df = df.sort_values("movieId")
    print(df.tail())
    cf = Collaborative_Filtering(100)
    cf.train(df)
    pred = cf.prediection(1, 1)
    print(pred)


if __name__ == "__main__":
    main()