# MEMBER

This is the official implementation of MEMBER **(Mixture-of-Experts for Multi-BEhavior Recommendation)** 

(Submission for ACM KDD 2025 Research Track)

---
 ### Datasets
We use three widely used datasets for multi-behavior recommendation, **Tmall**, **Taobao** and **Jdata**.

First, please preprocess the data in each data folder.
```bash
cd data/{data_name}

python preprocess.py
```
---
We have **three evaluation output results, overall performance under the standard evaluation, performance on the visited items, and performance on the unvisited items.**

### How to Run (Original Setting: Evaluation Protocol 1)
```bash
cd UniMBR
```
* **Tmall**
```bash
python main.py --data_name tmall --con 1e-2 --gen 1.0 --lambda_s 0.6 --neg_edge 3 --temp 0.5 --decay 1e-8 --setting ori --device [gpuid]
```
* **Taobao**
```bash
python main.py --data_name taobao --con 1e-3 --gen 1.0 --lambda_s 0.6 --neg_edge 3 --temp 0.7 --decay 1e-8 --setting ori --device [gpuid]
```
* **Jdata**
```bash
python main.py --data_name jdata --con 1e-4 --gen 1.5 --lambda_s 0.6 --neg_edge 5 --temp 0.7 --decay 1e-8 --setting ori --device [gpuid]
```

---

### Acknowledgement
This code is implemented based on the open source code from the paper **Behavior-Contextualized Item Preference Modeling for Multi-Behavior Recommendation** (SIGIR '24).

