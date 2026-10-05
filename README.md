# ML Zoomcamp – Module 02: Regression

My homework for Module 02 (Regression) of the DataTalksClub Machine Learning Zoomcamp, 2026 cohort. The notebook predicts a car's fuel efficiency (MPG) with linear regression that I wrote from scratch in NumPy.

## Dataset

The [Car Fuel Efficiency dataset](https://raw.githubusercontent.com/DataTalksClub/machine-learning-zoomcamp/main/cohorts/2026/data/car_fuel_efficiency_2026.csv), which the notebook loads straight from the course repo.

- **Features:** `engine_displacement`, `horsepower`, `vehicle_weight`, `model_year`
- **Target:** `fuel_efficiency_mpg`

## What the notebook covers

1. **EDA:** plots the distribution of the target and checks for missing values (`horsepower` has 877 missing; its median is 254).
2. **Train/validation/test split:** 60/20/20 with a seeded shuffle.
3. **Linear regression from scratch:** solves the normal equation, with an optional L2 (ridge) term.
4. **Missing-value strategy:** compares filling `horsepower` with 0 against filling it with the training-set mean.
5. **Regularization:** tries `r` values from 0 to 100.
6. **Seed sensitivity:** gets the standard deviation of validation RMSE across 10 random splits.
7. **Final model:** trains on train + validation and evaluates on the test set.

## Results

| Experiment | Validation RMSE |
|---|---|
| Fill missing with 0 | 2.205 |
| Fill missing with training mean | 2.202 |
| Best regularization (r = 0) | 2.2053 |
| Std. of RMSE across seeds 0–9 | 0.029 |
| **Final test RMSE** (seed 9, r = 0.001) | **2.236** |

## How to run

    pip install numpy pandas matplotlib seaborn jupyter
    jupyter notebook 02-Regression.ipynb

You need an internet connection because the dataset loads from GitHub.
