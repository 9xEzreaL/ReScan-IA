# ReScan-IA: A Spatially-Adaptive Diffusion Framework for Controllable 3D Intracranial Aneurysm Inpainting

---

## Overview

**ReScan-IA** is a spatially-adaptive 3D diffusion framework for controllable intracranial aneurysm (IA) synthesis in CTA volumes. We frame aneurysm insertion as a *virtual re-scan*: a clinically plausible aneurysm is introduced into an existing CTA volume as if it had been captured in a repeat acquisition, while preserving surrounding vascular anatomy.

The framework addresses two core challenges in IA data augmentation:
1. **Spatial controllability** — precise control over aneurysm location and morphology via decoder-level spatial modulation (3D SPADE).
2. **Anatomical consistency** — vessel-aware conditioning to enforce vascular continuity during inpainting.

<img src="images/overview.jpg" width="800" alt="ReScan-IA Framework Overview"/>

**(a)** Training procedure with vessel-aware and spatial conditioning. **(b)** Spatially-adaptive feature modulation mechanism. **(c)** Downstream synthetic data generation and mask sampling strategy during inference.

---

## Key Results

### Quantitative Performance — External Dataset A & B

Segmentation (Dice) and detection (FP/case, Precision, Recall) on two independent external cohorts. Case-wise Dice reported with 95% CI. Best results per dataset in **bold**.

**External Dataset A**

| Method | Voxel Dice (%) | Case Dice (%) [95% CI] | FP / Case [95% CI] | Precision (%) [95% CI] | Recall (%) [95% CI] |
|---|:---:|:---:|:---:|:---:|:---:|
| Real-10 | 35.90 | 10.24 [4.10–16.38] | 1.93 [1.42–2.44] | 9.26 [3.36–15.16] | 30.00 [16.26–43.74] |
| Synthetic-250 | 42.83 | 21.95 [15.45–28.45] | 11.71 [10.50–12.92] | 5.39 [4.12–6.67] | **88.89** [79.34–98.44] |
| Real-250 | 41.11 | 11.17 [4.82–17.51] | 5.07 [4.49–5.66] | 3.81 [1.73–5.88] | 30.00 [16.26–43.74] |
| **Real+Synthetic (250+250)** | **62.24** | **24.65** [16.27–33.03] | **1.83** [1.49–2.16] | **19.02** [12.22–25.82] | 58.89 [44.11–73.67] |

**External Dataset B**

| Method | Voxel Dice (%) | Case Dice (%) [95% CI] | FP / Case [95% CI] | Precision (%) [95% CI] | Recall (%) [95% CI] |
|---|:---:|:---:|:---:|:---:|:---:|
| Real-10 | 19.46 | 9.88 [4.39–15.37] | **1.72** [1.15–2.28] | 12.81 [5.35–20.28] | 27.78 [14.36–41.20] |
| **Synthetic-250** | **35.20** | **18.80** [13.12–24.49] | 12.16 [11.30–13.03] | 5.15 [3.93–6.36] | **87.04** [77.32–96.76] |
| Real-250 | 28.58 | 12.00 [6.47–17.53] | 5.28 [4.67–5.90] | 4.85 [2.61–7.08] | 38.89 [24.61–53.17] |
| Real+Synthetic (250+250) | 28.06 | 16.57 [9.48–23.67] | 1.82 [1.51–2.14] | **18.08** [11.17–24.98] | 51.11 [36.26–65.96] |

### Qualitative Ablation — Image Generation Quality

<img src="images/ablation.jpg" width="800" alt="Qualitative Ablation"/>

*From left to right: real CTA, aneurysm mask, Palette baseline, ReScan-IA w/o spatial modulation, and full ReScan-IA. Palette fails to preserve vascular continuity; removing spatial modulation leads to poor morphological control. ReScan-IA achieves anatomically consistent and controllable synthesis.*

### Qualitative Ablation — Downstream Segmentation

<img src="images/ablation_seg.jpg" width="800" alt="Downstream Segmentation Comparison"/>

*Synthetic-only training increases sensitivity but yields more false positives; real+synthetic training improves boundary accuracy and suppresses spurious detections.*

---


## Acknowledgements

This codebase is built upon the [Palette: Image-to-Image Diffusion Models](https://arxiv.org/pdf/2111.05826.pdf) implementation by [Janspiry](https://github.com/Janspiry/Palette-Image-to-Image-Diffusion-Models), extended substantially for 3D volumetric medical image synthesis with dual conditioning and spatially-adaptive feature modulation.

Our work builds on the following:

**Theoretical foundations:**
- [Denoising Diffusion Probabilistic Models (Ho et al., NeurIPS 2020)](https://arxiv.org/pdf/2006.11239.pdf)
- [Palette: Image-to-Image Diffusion Models (Saharia et al., SIGGRAPH 2022)](https://arxiv.org/pdf/2111.05826.pdf)
- [Diffusion Models Beat GANs on Image Synthesis (Dhariwal & Nichol, NeurIPS 2021)](https://arxiv.org/abs/2105.05233)
- [Semantic Image Synthesis with Spatially-Adaptive Normalization / SPADE (Park et al., CVPR 2019)](https://arxiv.org/abs/1903.07291)

**Code references:**
- [openai/guided-diffusion](https://github.com/openai/guided-diffusion)
- [Janspiry/Palette-Image-to-Image-Diffusion-Models](https://github.com/Janspiry/Palette-Image-to-Image-Diffusion-Models)

---

## License

This project is released under the [MIT License](LICENSE).