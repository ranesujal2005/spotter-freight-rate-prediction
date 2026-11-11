# Freight Rate Prediction Challenge

## Reproduce
```bash
python -m pip install -r requirements.txt
python train_predict.py
python score.py --predictions validation_predictions.csv --december-predictions data/december_chart_inputs.csv
```

The training script uses a chronological holdout (earliest 80% of unique dates for training, latest 20% for validation), selects CatBoost iterations by early stopping, then refits on all 48,000 labelled records. It creates `validation_predictions.csv` and fills the December input file.

## Files
- `data/train_test.csv`: labelled development data
- `data/validation.csv`: final prediction rows
- `data/validation_predictions_template.csv`: required ID template
- `data/december_chart_inputs.csv`: completed 31-day chart inputs
- `validation_predictions.csv`: final 12,000 predictions
- `catboost_model.cbm`: fitted model from this run

## Validation
Chronological cutoff: 2025-09-01. Holdout: 9,523 rows. Training: 38,477 rows.
Initial holdout metrics: MAE $113.60, RMSE $632.87, R² 0.8280.
