# AdverScan — Dataset Instructions

## Currently included

`sample_data.csv` — 500-row synthetic dataset in CICIDS2017 format.
* 300 rows **BENIGN** (typical web browsing traffic)
* 100 rows **DoS Hulk** (HTTP flood attack)
* 100 rows **Web Attack** (SQL injection / XSS patterns)

This file lets you run `python run_experiment.py` immediately without any download.
Results are for demonstration purposes only.

---

## Upgrading to the real CICIDS2017 dataset

### Step 1 — Download

Go to: https://www.unb.ca/cic/datasets/ids-2017.html

Download file: **Wednesday-workingHours.pcap_ISCX.csv** (~450 MB)

### Step 2 — Place the file

Save it in this folder and rename it:

```
adverscan/data/cicids2017_wednesday.csv
```

### Step 3 — Re-run

```bash
python run_experiment.py
```

The pipeline auto-detects the full dataset and uses it automatically.

---

## Why Wednesday?

The Wednesday capture contains the heaviest attack traffic in CICIDS2017:
- DoS Hulk, DoS GoldenEye, DoS Slowloris, DoS SlowHTTPTest
- Heartbleed (port 444)

This makes it the most interesting day for adversarial ML experiments.

## Column format (CICIDS2017)

The CSV has 79 columns (78 numeric features + `Label`).  
The pipeline strips leading/trailing whitespace from column names automatically.

Label values in the full dataset:
```
BENIGN, DoS Hulk, DoS GoldenEye, DoS Slowloris, DoS SlowHTTPTest, Heartbleed
```
All non-BENIGN labels are treated as `attack = 1` (binary classification).
