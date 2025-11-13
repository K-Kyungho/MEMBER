# MEMBER

This is the official implementation of MEMBER **(Mixture-of-Experts for Multi-BEhavior Recommendation)** 

[![arXiv](https://img.shields.io/badge/arXiv-2508.19507-b31b1b.svg)](https://arxiv.org/abs/2508.19507)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![Python](https://img.shields.io/badge/python-3.8+-blue.svg)

(Accepted for ACM CIKM 2025 Full Research Paper)

---

## 📑 Paper & Appendix

- Paper: [arXiv:2508.19507](https://arxiv.org/abs/2508.19507)  
- Accepted at: **ACM CIKM 2025 (Full Research Paper)**  
- Online Appendix: See `MEMBER-Online Appendix.pdf` in the root folder.

---
## 📊 Datasets

We use three widely adopted multi-behavior recommendation datasets:  
- **Tmall**  
- **Taobao**  
- **Jdata**

### Preprocessing
```bash
cd data/{data_name}
python preprocess.py
```

---
## 📈 Evaluation

We report three evaluation results:
1. **Overall performance** under the standard evaluation  
2. **Performance on visited items**  
3. **Performance on unvisited items**
   
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

## 📚 Citation
If you find MEMBER useful, please cite our paper:
```bibtex
@inproceedings{kim2025self,
  title     = {A Self-Supervised Mixture of Experts Framework for Multi-behavior Recommendation},
  author    = {Kim, Kyungho and Kim, Sunwoo and Lee, Geon and Shin, Kijung},
  booktitle = {CIKM},
  year      = {2025}
}
