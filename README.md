# GermEval24

## Installation
```pip install .```

## Model Training

```bash
sh train_scripts/m1.sh
sh train_scripts/m2.sh
sh train_scripts/m3.sh
sh train_scripts/m4.sh
sh train_scripts/m5.sh
sh train_scripts/m6.sh
sh train_scripts/m7.sh
```

## Prediction

After training the model paths must be defined in `scripts/prediction.py [35:43]`.

```bash
# Subtask 1, Approach 1
python scripts/prediction.py --output st1_predictions.tsv --approach 1

# Subtask 1, Approach 2
python scripts/prediction.py --output st1_predictions.tsv --approach 2

# Subtask 2
python scripts/prediction.py --output st2_predictions.tsv
```
