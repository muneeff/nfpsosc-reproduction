import os

# 1. تحديث أو إنشاء ملف README.md بالنسخة المحدثة للنموذج DR-NFPSO
readme_content = """# DR-NFPSO: Dynamic Radius-Adaptive Neuro-Fuzzy Particle Swarm Optimization

## Overview
This repository provides the official implementation of the **Dynamic Radius-Adaptive Neuro-Fuzzy Particle Swarm Optimization (DR-NFPSO)** framework, as described in our paper: 
*"Dynamic Radius-Adaptive Neuro-Fuzzy Particle Swarm Optimization for Sparse Time-Series Forecasting: A Self-Tuning Explainable Framework"* (Published in *Engineering Applications of Artificial Intelligence*).

## Key Features
- **Dynamic Radius Co-Optimization:** Simultaneous tuning of subtractive clustering radius ($r_c$) and Gaussian antecedent parameters within the PSO loop.
- **Pure-Residual Hybrid Strategy:** Mathematical decomposition of raw series into deterministic linear trends and stochastic nonlinear residuals for improved forecasting stability.
- **Leakage-Aware Protocol:** Validation framework ensuring no information leakage across 8,780 experimental tasks.
- **Parsimonious Rule Base:** Automated active-mask mechanism ($R \\le 2$) for structural regularization in sparse data conditions ($N < 25$).

## Folder Structure
- `/src`: Core implementation of the DR-NFPSO algorithm.
- `/data`: Benchmark synthetic datasets and real-world short-series samples.
- `/scripts`: Execution scripts for the Development-Freeze-Final protocol.
- `/models`: Pre-trained models and consequent solver (Ridge regression).

## Citation
If you use this code in your research, please cite our paper:
*Ahmed, M. A. F. (2026). Dynamic Radius-Adaptive Neuro-Fuzzy Particle Swarm Optimization for Sparse Time-Series Forecasting. Engineering Applications of Artificial Intelligence.*
"""

with open("README.md", "w", encoding="utf-8") as f:
    f.write(readme_content)

print("تم تحديث ملف README.md بنجاح.")

# 2. إنشاء هيكل المجلدات الأساسية إذا لم تكن موجودة
folders = ["src", "data", "scripts", "models"]
for folder in folders:
    os.makedirs(folder, exist_ok=True)
    print(f"تم التأكد من وجود المجلد: {folder}/")

print("\nالخطوات التالية لتحديث المستودع على جيت هب (نفذها في التيرمينال):")
print("git add .")
print('git commit -m "Update repository with DR-NFPSO framework and EAAI paper details"')
print("git push origin main")