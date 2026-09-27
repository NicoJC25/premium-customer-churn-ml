# Model card

## Model overview

- **Task:** binary classification of customer churn.
- **Population:** customers in the top revenue quartile of the supplied telecom cohort.
- **Primary metric:** ROC-AUC on an untouched stratified holdout set.
- **Baseline:** dummy classifier using the class prior.
- **Candidates:** logistic regression, random forest, and XGBoost.
- **Selection:** highest holdout ROC-AUC after training-only model selection.

## Inputs

Numerical and categorical account/usage attributes after removing identifiers,
the target, sensitive fields, high-missingness columns, and constant columns.
Missing values and categorical encoding are fitted inside each model pipeline,
using training data only.

## Output

A probability-like churn risk score. The default 0.5 threshold is used for the
reported classification metrics. Any operational threshold should instead be
chosen using intervention costs and benefits, then validated prospectively.

## Evaluation

See `reports/metrics.json` and `reports/model_comparison.csv` for the latest
aggregate run. These artifacts contain no row-level data.

## Limitations

See [Responsible AI and limitations](responsible-ai.md). The model is a portfolio
proof of concept and is not approved for production use.
