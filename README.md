# CIF4SL


## Environment

Use Python 3.10 or later and a PyTorch installation compatible with your CPU or CUDA environment. From the project root:

```bash
python -m pip install -r requirements.txt
```


## Run

Processed inputs are included under `data/slbenchmark/processed/` and `data/perturbation_responses/processed/`. All experiment settings are loaded from `configs/cif4sl.yaml`.

Run five-fold training and evaluation:

```bash
python -m cif4sl run --config configs/cif4sl.yaml
```

Run one fold on a selected device:

```bash
python -m cif4sl run --fold-ids 0 --device cuda --output results/fold0
```

Use `--device cpu` when CUDA is unavailable. 



## Reproduction

The default configuration retains seed 123, five-fold CV1 evaluation, random negatives at 1:1 and a 10% inner validation fraction. Training uses 300 epochs. Both epochs optimize response, margin and Bliss losses; joint checkpoint selection uses validation NDCG@10. Test pairs are excluded from the SL propagation graph and response supervision.

Ranking uses the full gene catalog, excluding self-pairs and training/validation pairs. The classification threshold is selected on validation data. Results are written to `results/cif4sl/fold_metrics.csv` and `summary.csv` (mean and sample standard deviation across folds). Completed folds are reused automatically; use a new output directory after changing code, data or configuration.

