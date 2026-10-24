# AdverScan — Adversarial ML Security Lab

> An educational experiment showing how adversarial perturbations can reduce the effectiveness of machine-learning intrusion-detection systems and how defensive techniques can improve resilience.

## What this project does

AdverScan trains an intrusion-detection classifier on network-traffic-style data, attacks it through transferable adversarial examples, then compares two defenses. The project is designed to make adversarial machine-learning concepts measurable through a reproducible experiment, plots, and interactive notebooks.

## Core functions

- Prepare synthetic or CICIDS2017-style tabular network data.
- Train Random Forest and Logistic Regression baseline models.
- Create FGSM and PGD-style perturbations using a differentiable Logistic Regression surrogate.
- Evaluate whether attacks transfer to the Random Forest target model.
- Compare baseline performance with adversarial training and feature squeezing.
- Produce metrics, confusion matrices, and robustness plots.

## Architecture

```text
Network-traffic dataset
  │
  ▼
Data preparation → target IDS model + surrogate model
  │
  ▼
FGSM / PGD perturbation generator
  │
  ▼
Transfer attack evaluation on IDS target
  │
  ▼
Defenses: adversarial training | feature squeezing
  │
  ▼
Metrics, plots, notebooks, and experiment report
```

## Technology

- Python
- scikit-learn, NumPy, and Pandas
- Matplotlib and Seaborn
- Jupyter notebooks
- Optional PyTorch/Foolbox integrations

## Run locally

```bash
cd adverscan
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt

python run_experiment.py
jupyter notebook
```

## Repository layout

```text
adverscan/
├── src/         # Data, model, attack, defense, experiment, visualisation code
├── notebooks/   # Step-by-step exploratory and reporting notebooks
├── data/        # Sample data and full-dataset instructions
├── results/     # Locally generated figures and metrics
└── run_experiment.py
```

## Responsible-use note

This repository is for defensive research, education, and robustness testing on owned or authorized models. It demonstrates model evasion so defenders can measure and improve detection systems; it must not be used to bypass production security controls.
