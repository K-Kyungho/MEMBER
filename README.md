# UniMBR

This is the official implementation of UniMBR **(Unified framework for Mutil-Behavior Recommendation)** 

(Submission for ACM WWW 2025 Research Track)

---
 ### Datasets
We use three widely used datasets for multi-behavior recommendation, **Tmall**, **Taobao** and **Jdata**.

First, please preprocess the data in each data folder.
```bash
cd data/{data_name}
python preprocess.py
```
---
### How to Run (Original Setting: Evaluation Protocol 1)

```bash
cd UniMBR
```
* **Tmall**
```bash
python main.py --data_name Tmall --con 1e-2 --gen 1.0 --lambda_s 0.6 --neg_edge 3 --temp 0.5 --decay 1e-8 --setting ori --device [gpuid]
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
### How to Run (New Setting (Proposed): Evaluation Protocol 2)

```bash
cd UniMBR
```
* **Tmall**
```bash
python main.py --data_name Tmall --con 0.1 --gen 1.0 --lambda_s 0.5 --neg_edge 3 --temp 0.3 --decay 1e-7 --setting new --device [gpuid]
```

* **Taobao**
```bash
python main.py --data_name taobao --con 0.1 --gen 1.0 --lambda_s 0.5 --neg_edge 3 --temp 0.5 --decay 1e-7 --setting new --device [gpuid]
```

* **Jdata**
```bash
python main.py --data_name jdata --con 0.05 --gen 0.5 --lambda_s 0.8 --neg_edge 3 --temp 1.0 --decay 1e-8 --setting new --device [gpuid]
```

---
### Acknowledgement
This code is implemented based on the open source code from the paper **Behavior-Contextualized Item Preference Modeling for Multi-Behavior Recommendation** (SIGIR '24).

