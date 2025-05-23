# MEMBER

This is the official implementation of MEMBER **(Mixture-of-Experts for Multi-BEhavior Recommendation)** 

(Submission for ACM CIKM 2025 Full Research Papers)
---
---
### Online Appendix
For additional details, please refer to the online appendix of MEMBER.

[Online Appendix.pdf](https://github.com/user-attachments/files/20407967/_CIKM_2025__Multi_relational_Recommendation__Online_Appendix_.pdf)


---
 ### Datasets
We use three widely used datasets for multi-behavior recommendation, **Tmall**, **Taobao** and **Jdata**.

First, please preprocess the data in each data folder.
```bash
cd data/{data_name}

python preprocess.py
```
---
We have **three evaluation results: (1) overall performance under the standard evaluation, (2) performance on the visited items, and (3) performance on the unvisited items.**

### How to Run MEMBER
```bash
cd METHOD
```
* **Tmall**
```bash
python main.py --data_name tmall --con_s 0.1 --temp_s 0.6  --con_us 0.1 --temp_us 0.7 --gen 0.1 --lambda_s 0.6 --alpha 2
```
* **Taobao**
```bash
python main.py --data_name taobao --con_s 0.1 --temp_s 0.8 --con_us 0.1 --temp_us 0.7 --gen 0.1 --lambda_us 0.6
```
* **Jdata**
```bash
python main.py --data_name jdata --con_s 0.1 --temp_s 0.6 --con_us 0.01 --temp_us 1.0 --gen 0.01 --lambda_s 0.4 --lambda_us 0.4 --alpha 2
```

---

### Acknowledgement
This code is implemented based on the open source code from the paper **Behavior-Contextualized Item Preference Modeling for Multi-Behavior Recommendation** (SIGIR '24).

