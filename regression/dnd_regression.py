import numpy as np
import ml_lib
import pandas as pd
from sklearn.model_selection import train_test_split


# Generate a string version of the formula for regression (right-hand side only)
def generate_formula(beta, labels):
    formula_string = ""
    if beta[0] != 0:
        formula_string += f"{beta[0]:.3f}"
    for i in range(1, len(beta)):
        if beta[i] != 0:
            if formula_string != "":
                formula_string += " + "
            formula_string += f"{beta[i]:.3f} * {labels[i - 1]}"
    return formula_string


# Read in the csv file with appropriate columns
dnd_monsters = pd.read_csv("dnd_monsters.csv", usecols=['cr', 'ac', 'hp', 'str', 'dex', 'con', 'int', 'wis', 'cha'])
dnd_monsters.dropna(inplace=True)

# The features that we will be using to predict challenge rating ("cr")
features = ['ac', 'hp', 'str', 'dex', 'con', 'int', 'wis', 'cha']

# Some of the challenge rating entries are fractions so eval fixes that.
# challenge_rating is the dependent variable
dnd_monsters["cr"] = dnd_monsters["cr"].apply(eval)
challenge_rating = dnd_monsters["cr"].to_numpy()
print(dnd_monsters.describe())

# The monster_stats holds the data for the features (independent variables)
monster_stats = dnd_monsters.loc[:, features].to_numpy()

# Generate test/train split
stats_train, stats_test, challenge_train, challenge_test = train_test_split(monster_stats, challenge_rating, test_size=0.3)

# Multiple linear regression with the whole dataset:
print("=" * 160)
print("Multiple linear regression for the entire dataset (401 monsters)")
multiple_linear = ml_lib.MultipleLinear(monster_stats, challenge_rating)
print(f"cr           = {generate_formula(multiple_linear.beta, features)}")
print(f"R^2          = {multiple_linear.R2_train:.3f}")
print(f"RMSE         = {multiple_linear.RMSE_train:.3f}")

# Multiple linear regression for the train/test split at 70/30
print("=" * 160)
print("Multiple linear regression for train/test split (70/30)")
multiple_linear = ml_lib.MultipleLinear(stats_train, challenge_train)
print(f"cr           = {generate_formula(multiple_linear.beta, features)}")
print(f"R^2 (train)  = {multiple_linear.R2_train:.3f}")
print(f"RMSE (train) = {multiple_linear.RMSE_train:.3f}")
print(f"R^2 (test)   = {multiple_linear.R2(stats_test, challenge_test):.3f}")
print(f"RMSE (test)  = {multiple_linear.RMSE(stats_test, challenge_test):.3f}")

# Ridge regression with the whole dataset with alpha = 1.0:
print("=" * 160)
print("Ridge regression for the entire dataset (401 monsters) with alpha = 1.0")
ridge = ml_lib.Ridge(monster_stats, challenge_rating)
print(f"cr           = {generate_formula(ridge.beta, features)}")
print(f"R^2          = {ridge.R2_train:.3f}")
print(f"RMSE         = {ridge.RMSE_train:.3f}")

# Ridge regression for the train/test split at 70/30 with alpha = 1.0
print("=" * 160)
print("Ridge regression for train/test split (70/30) with alpha = 1.0")
ridge = ml_lib.Ridge(stats_train, challenge_train)
print(f"cr           = {generate_formula(ridge.beta, features)}")
print(f"R^2 (train)  = {ridge.R2_train:.3f}")
print(f"RMSE (train) = {ridge.RMSE_train:.3f}")
print(f"R^2 (test)   = {ridge.R2(stats_test, challenge_test):.3f}")
print(f"RMSE (test)  = {ridge.RMSE(stats_test, challenge_test):.3f}")

# Lasso regression with the whole dataset with alpha = 1.0:
print("=" * 160)
print("Lasso for the entire dataset (401 monsters) with alpha = 1.0")
lasso = ml_lib.Lasso(monster_stats, challenge_rating)
print(f"cr           = {generate_formula(lasso.beta, features)}")
print(f"R^2          = {lasso.R2_train:.3f}")
print(f"RMSE         = {lasso.RMSE_train:.3f}")

# Lasso regression for the train/test split at 70/30 with alpha = 1.0
print("=" * 160)
print("Lasso for train/test split (70/30) with alpha = 1.0")
lasso = ml_lib.Lasso(stats_train, challenge_train)
print(f"cr           = {generate_formula(lasso.beta, features)}")
print(f"R^2 (train)  = {lasso.R2_train:.3f}")
print(f"RMSE (train) = {lasso.RMSE_train:.3f}")
print(f"R^2 (test)   = {lasso.R2(stats_test, challenge_test):.3f}")
print(f"RMSE (test)  = {lasso.RMSE(stats_test, challenge_test):.3f}")
print("=" * 160)


