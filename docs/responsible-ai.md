# Responsible AI and limitations

This project is an educational proof of concept, not a production decision
system. Churn scores should support human review and controlled experiments;
they should not automatically determine pricing, eligibility, or access to a
service.

## Privacy

- The original CSV files and row-level predictions are excluded from Git.
- `Customer_ID` is used only to merge the source tables and is never a model
  feature.
- Aggregate metrics and plots contain no customer identifiers.

## Sensitive and proxy attributes

The source data includes demographic, household, geographic, and credit-related
attributes. The training pipeline excludes them by default, including ethnicity,
income, marital status, household composition, geography, and credit indicators.
This reduces discriminatory targeting risk but does not prove fairness: remaining
variables can still act as proxies.

Before a real deployment, the model would require:

1. A documented lawful purpose and data-retention policy.
2. Group-level error-rate and calibration audits with appropriately governed data.
3. Review for proxy discrimination and accessibility impacts.
4. Human appeal and override mechanisms.
5. Monitoring for data drift, performance decay, and offer-allocation bias.

## Statistical limitations

- The dataset represents one historical cohort from an anonymous provider.
- The analysis is observational; feature importance is not evidence of causality.
- ROC-AUC measures ranking ability, not financial impact.
- Business impact must be established through a prospective randomized experiment.
- The revenue examples in the original course presentation were illustrative and
  are intentionally not reported as validated savings in this repository.

## Intended use

Appropriate: education, portfolio review, pipeline demonstration, and hypothesis
generation.

Inappropriate: automated customer treatment, lending or credit decisions, pricing,
or any high-impact decision without additional governance and validation.
