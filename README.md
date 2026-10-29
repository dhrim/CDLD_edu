# Educational response representations

Numerical reproduction for **Task- and predictor-dependent performance of educational response representations: Item classification and response correctness prediction in EdNet and Eedi**.

## View the results

Open **[paper_results.ipynb](paper_results.ipynb)** to inspect the executed tables, paired confidence intervals, domain confusion matrices, run-level metrics and figure directly on GitHub. No installation is required to read its saved outputs.

## Reproduce

Use Python 3.10–3.12 on a CPU. Tested with Python 3.10.6 and the pinned numerical dependencies.

```bash
python -m pip install -r requirements.txt
python -m pip install notebook
jupyter notebook paper_results.ipynb
```

Select **Run All**. The notebook calls `reproduce.py` and displays its calculated outputs. Alternatively:

```bash
python reproduce.py
```

The script recalculates metrics and 5,000 paired bootstrap resamples from `predictions/`, checks them against `expected.json` with absolute and relative tolerances of 1e-10, and writes JSON, CSV and Figure 1 (PDF/PNG) to `output/`. The reference file is used only for verification, not to calculate results.

## Contents

- `paper_results.ipynb`: executable analysis viewer with saved results
- `reproduce.py`: metric, bootstrap and figure calculations
- `predictions/`: compressed fitted predictions, labels, pairing indices and response counts
- `requirements.txt`: numerical dependencies
- `expected.json`: reference statistics for checking reproduction
- `manifest.json`: input checksums

The notebook covers item classification, bundle-grouped classification, correctness prediction, between-run variation, evaluation eligibility, response-count distributions and Eedi domain-level results. B denotes IRT estimates and response summaries. Smaller log loss and larger AUC, balanced accuracy and macro-F1 indicate better performance.

## Reproduction scope

This package reproduces numerical analyses **from saved predictions**. It does not train the representation models or downstream predictors. No GPU or raw dataset download is needed. It contains neither the full original response matrices nor personal names or contact details.

Classification bootstrap units are item IDs or bundle IDs; correctness bootstrap units are student IDs. The same ID receives the same multiplicity across runs. Confidence intervals are conditional on the fitted models. Correctness fits with and without the item-rate input are separate fits, so their differences do not isolate a causal input effect.

Some multiclass probabilities have small floating-point row-sum deviations. The pinned scikit-learn version can emit a probability-sum warning while reproducing the reference metrics; saved probabilities are not modified.

Source datasets: [EdNet](https://github.com/riiid/ednet) and [Eedi NeurIPS 2020 Education Challenge](https://www.eedischool.com/projects/neurips-education-challenge). Obtain source data from their providers under their terms.
