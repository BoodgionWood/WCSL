# WCSL

This is the official code implementation for the TMLR paper  
**_Wasserstein Coreset via Sinkhorn Loss_**.

GitHub: [BoodgionWood/WCSL](https://github.com/BoodgionWood/WCSL)

## 📢 Update

We have released the code for reproducing **Figure 4** and **Figure 9** from the paper.  
More experiments and modules are being cleaned and will be released soon.

---

## 🔧 Reproducing Results

### 🔹 Figure 4 (Synthetic Distributions)

1. Navigate to the directory:
    ```bash
    cd experiments/distance
    ```

2. Run the following commands:
    ```bash
    python distance.py
    python Figure_4or9_distance.py
    ```

---

### 🔹 Figure 9 (Image Datasets)

1. Navigate to the directory:
    ```bash
    cd experiments/distance
    ```

2. Run the following commands:
    ```bash
    python distance.py --data_type="image" --download
    python Figure_4or9_distance.py --data_type="image"
    ```

---

## 📦 Required Packages

The following Python packages are used in this repository:

- `numpy`
- `torch` (version: 2.7.1+cu128)
- `scipy`
- `matplotlib`
- `seaborn`
- `pandas`
- `pathlib`

We recommend using Python ≥3.8 and a virtual environment for package management.

---

## 📁 Repository Structure (WIP)

- `experiments/distance/`: Scripts to compute distance-based metrics and generate key figures.
- Additional modules and scripts for reproducing the full set of experiments will be released incrementally.

---

## 📄 Citation

If you find our work useful, please consider citing:

```bibtex
@article{yin2025wasserstein,
  title={Wasserstein Coreset via Sinkhorn Loss},
  author={Haoyun Yin and Yixuan Qiu and Xiao Wang},
  journal={Transactions on Machine Learning Research},
  issn={2835-8856},
  year={2025},
  url={https://openreview.net/forum?id=DrMCDS88IL},
  note={}
}
```

---

For questions or issues, feel free to open an [issue](https://github.com/BoodgionWood/WCSL/issues) or reach out to the authors.
