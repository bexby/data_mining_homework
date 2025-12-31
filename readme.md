# 数据挖掘大作业
**Notes: 层次聚类和高斯混合模型代码运行部分无需安装任何库, 只需安装`python 3.10.18`, 画图部分需要安装`matplotlib`与`numpy`**

## 层次聚类
### 运行  
修改`HC.py`中的:
```
DATA_PATH = "./data/data.txt"
SAVE_PATH = "./result/HC/HC_log_max.jsonl"
DISTANCE_MODE = "max"
```
执行:
```
cd data_mining_homework
python HC.py
```
### 画图
修改`plot_HC.py`中的:
```
plot_hc_from_jsonl("./result/HC/HC_log_min.jsonl")
```
执行:
```
cd data_mining_homework/plot_code
python plot_HC.py
```

## 高斯混合模型
### 运行  
修改`GMM.py`中的:
```
K = 5
STEPS = 100
ACCELERATE = True
DATA_PATH = "./data/data.txt"
SAVE_PATH = f"./result/GMM/GMM_log_{K}_kmeans.jsonl"
INITIAL_METHOD = custom_initialize
```
执行:
```
cd data_mining_homework
python GMM.py
```
### 画图
修改`plot_HC.py`中的:
```
DATA_PATH = "./data/data.txt"
RESULT_PATH = "./result/GMM/GMM_log_15_kmeans_new.jsonl"
```
执行:
```
cd data_mining_homework/plot_code
python plot_GMM.py
```

## 基于物品的协同过滤
### 环境
```
numpy, pandas
```
### 运行  
修改`CF.py`中的:
```
df = pd.read_csv("./data/ml-latest-small/ratings.csv")
cf = CollaborativeFiltering(min_common_user=2)
pred = cf.prediection(1, 1) # 预测用户1对电影1的评分
```
执行:
```
cd data_mining_homework
python GMM.py
```
### K折交叉验证
示例:
```
python ck_kfold.py --ratings "./data/ml-latest-small/ratings.csv"
```