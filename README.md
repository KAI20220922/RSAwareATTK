# RSAwareATTK  
**Official implementation of the attack proposed in:**  
*A Targeted Attack against Certifiably Robust Smoothed Classifiers*  
Kai Zeng, Mauro Barni, Benedetta Tondi  
IEEE International Workshop on Information Forensics and Security (WIFS), 2025

---

## Overview

This repository contains the official implementation of the **RS-aware targeted attack** introduced in the paper:

> **A Targeted Attack against Certifiably Robust Smoothed Classifiers**

The provided implementation corresponds exactly to the **targeted attack against the smoothed classifier** used in the paper experiments.

The attack explicitly accounts for the randomized smoothing prediction mechanism and performs optimization accordingly.

---

# Usage

The following snippet reproduces the **targeted RS-aware attack** used in the paper *A Targeted Attack against Certifiably Robust Smoothed Classifiers* (WIFS 2025):

```python
attacker = RSAwareATTK(
    steps=args.num_steps,
    device=device,
    temperature=args.temperature,
    max_norm=args.epsilon
)

x_adv = attacker.attack_whitebox(
    smoothedmodel=smoothed_classifier,
    basemodel=base_classifier,
    inputs=x,
    labels=label,
    N=args.N,
    alpha=args.alpha,
    batch=args.batch,
    noise=noise,
    num_noise_vectors=args.num_noise_vec,
    no_grad=args.no_grad_attack,
    use_clip_range=False
)
```
---
# Parameters

## `RSAwareATTK` Constructor

- **steps**  
  Number of optimization iterations.

- **device**  
  Computation device (e.g., `torch.device("cuda")`).

- **temperature**  
  Temperature parameter used in the logistic transformation of softmax outputs.  
  It controls the sharpness of the RS-aware objective.

- **max_norm**  
  In the paper, the white-box attack does **not** explicitly enforce an L2 norm constraint.  
  In this implementation, `max_norm` is used to control the **step size (learning rate)** of the optimizer, rather than acting as a hard perturbation bound.
  In the white-box attack used in the paper, the step size (denoted as **α** in the paper) is implemented through the SGD learning rate.
  To keep the attack strength consistent with the experimental setting, the learning rate is computed as:

  ```python
  lr = self.max_norm / 400 * 2
  ```

---

## `attack_whitebox(...)` Arguments

- **smoothedmodel**  
  Instance of the `Smooth` classifier implementing randomized smoothing.

- **basemodel**  
  Underlying base neural network.

- **inputs**  
  Input tensor. Must follow the same normalization used for the base model.

- **labels**  
  True labels (untargeted attack) or target labels (targeted attack).

- **N**  
  Number of Monte Carlo samples used for smoothed prediction.

- **alpha**  
  Confidence level used in randomized smoothing certification.

- **batch**  
  Internal batch size for Monte Carlo sampling.

- **noise**  
  Optional Gaussian noise tensor consistent with the smoothing procedure.

- **num_noise_vectors**  
  Number of noise realizations jointly optimized per input.

- **no_grad**  
  Reserved argument (kept for compatibility).

- **use_clip_range**  
  If `True`, clipping is applied in normalized input space (e.g., ImageNet bounds).

---

# Controlling the Attack Strength

The strength of the RSAware white-box attack can be adjusted through the following parameters:

---

## 1️⃣ Step Size (α)

The effective step size (denoted as **α** in the paper) is controlled through the `max_norm` parameter:

```python
lr = self.max_norm / 400 * 2
```
## 2️⃣ Number of Optimization Steps (`steps`)

- Increasing **`steps`** (corresponding to **T** in the paper) allows more optimization iterations.
- More iterations generally increase attack success rate.
- Larger `steps` typically produce stronger but slower attacks.

---

## 3️⃣ Number of Noise Vectors (`num_noise_vectors`)

- Increasing **`num_noise_vectors`** improves the approximation of the smoothed classifier.
- Larger values make the attack more aligned with the randomized smoothing decision rule.
- This typically leads to stronger but computationally more expensive attacks.

---

## Practical Guidance

To strengthen the attack:

- Increase `max_norm` (larger step size α)
- Increase `steps`
- Increase `num_noise_vectors`

To weaken the attack:

- Decrease `max_norm`
- Reduce `steps`
- Reduce `num_noise_vectors`

Adjust these parameters depending on the desired trade-off between attack strength and computational cost.
---

## Reference

If you use this code, please cite:

```bibtex
@inproceedings{zengwifs2025,
  title={A Targeted Attack against Certifiably Robust Smoothed Classifiers},
  author={Kai Zeng, Mauro Barni and Benedetta Tondi},
  booktitle={IEEE International Workshop on Information Forensics and Security},
  year={2025}
}
