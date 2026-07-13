# Stable Recommendation Mechanisms

Code and simulation results for the paper **Stable Recommendation Mechanisms**, by Omer Madmon and Moshe Tennenholtz.

## Repository Structure

```text
code/          Source code and scripts for simulations, verification, and plotting
data/          CSV files used to generate the final figures
figures/       Final paper figures
requirements.txt
README.md
```

The repository is self-contained. The included CSV files are the simulation outputs used to generate the figures in `figures/`.

## Setup

Create and activate a Python environment, then install the dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Quick Smoke Test

Run a small BRD/NRD sanity check:

```bash
python code/run_smoke.py
```

## Reproduce Figures from Included Data

The final figures can be regenerated from the CSV files in `data/`:

```bash
python code/plot_suite.py \
  --kind paper-ready-between \
  --brd-input data/between_family_brd_100_seed.csv \
  --nrd-input data/between_family_nrd_100_seed.csv \
  --output-dir figures

python code/plot_suite.py \
  --kind paper-ready-hyperparameters \
  --brd-input data/hyperparameters_brd_100_seed.csv \
  --nrd-input data/hyperparameters_nrd_100_seed.csv \
  --output-dir figures
```

These commands produce the seven final figures:

```text
sweep_lambda.pdf
sweep_n.pdf
sweep_s.pdf
sweep_k.pdf
hyperparameter_linear.pdf
hyperparameter_root.pdf
hyperparameter_logarithmic.pdf
```

PNG versions are generated as well.

## Verify Included Data

The main result files use 100 seeds per plotted point. To verify the data:

```bash
python code/verify_results.py --input data/between_family_brd_100_seed.csv --min-seeds 100 --target average
python code/verify_results.py --input data/between_family_nrd_100_seed.csv --min-seeds 100 --target average
python code/verify_results.py --input data/hyperparameters_brd_100_seed.csv --min-seeds 100 --target average
python code/verify_results.py --input data/hyperparameters_nrd_100_seed.csv --min-seeds 100 --target average
```

BRD convergence is measured in last-iterate sense. NRD convergence is measured using the average profile, matching the no-regret guarantee.

## Rerun Simulations

The included CSVs are already sufficient to reproduce the figures. To rerun the simulations, use `code/research_suite.py`. For example:

```bash
python code/research_suite.py \
  --suite between \
  --dynamics brd \
  --seeds 100 \
  --brd-rounds 1000 \
  --nrd-rounds 5000 \
  --nrd-check-every 100 \
  --workers 4 \
  --output data/between_family_brd_100_seed.csv

python code/research_suite.py \
  --suite between \
  --dynamics nrd \
  --seeds 100 \
  --brd-rounds 1000 \
  --nrd-rounds 5000 \
  --nrd-check-every 100 \
  --workers 4 \
  --output data/between_family_nrd_100_seed.csv

python code/research_suite.py \
  --suite hyperparameters \
  --dynamics brd \
  --seeds 100 \
  --brd-rounds 1000 \
  --nrd-rounds 5000 \
  --nrd-check-every 100 \
  --workers 4 \
  --output data/hyperparameters_brd_100_seed.csv

python code/research_suite.py \
  --suite hyperparameters \
  --dynamics nrd \
  --seeds 100 \
  --brd-rounds 1000 \
  --nrd-rounds 5000 \
  --nrd-check-every 100 \
  --workers 4 \
  --output data/hyperparameters_nrd_100_seed.csv
```

The runner resumes from existing output files by skipping jobs that are already present.
