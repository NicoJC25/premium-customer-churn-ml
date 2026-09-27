# Data access

The original dataset is intentionally **not included** in this repository.

It was supplied for the GCI World April 2026 final assignment and contains two
files from an anonymous telecommunications provider:

- `Client.csv`: customer and account attributes.
- `Record.csv`: usage history and the churn target.

The course documentation references strict data-protection constraints and
does not provide a public redistribution license. Do not commit either file,
course documents, row-level predictions, or customer identifiers.

## Local placement

Authorized users can place the files here:

```text
data/raw/Client.csv
data/raw/Record.csv
```

The paths can also be passed explicitly to the command-line interface.

## Public reproducibility

Run the synthetic-data generator to exercise the entire pipeline without the
private course data:

```bash
python scripts/generate_synthetic_data.py --output-dir data/synthetic
python -m churn_model.cli \
  --client data/synthetic/Client.csv \
  --record data/synthetic/Record.csv \
  --output-dir artifacts/synthetic \
  --quick
```

Synthetic data is generated only for software verification. Its scores must
not be compared with the validated aggregate results from the course dataset.
