import numpy as np
import ml_lib
import pandas as pd
import matplotlib.pyplot as plt


# Read in Pokemon data
pokemon = pd.read_csv("pokemon_alopez247.csv")
print(pokemon.describe())

"""
Simple linear regression on Pokemon data
Independent variable - Pokemon height in meters
Dependent variable - Pokemon weight in kilograms
"""
height = pokemon["Height_m"].to_numpy()
weight = pokemon["Weight_kg"].to_numpy()

# Perform simple linear regression
simple_linear = ml_lib.SimpleLinear(height, weight)

# Output results
print("\nSimple linear regression predicting Pokemon weight (kg) from height (m).")
print(f"Weight = {simple_linear.beta[0]:.3f} + {simple_linear.beta[1]:.3f} * height")
print(f"R^2    = {simple_linear.R2_train:.3f}")
print(f"RMSE   = {simple_linear.RMSE_train:.3f}")

# Plot results
simple_linear.plot()

"""
Multiple linear regression on Pokemon data
Independent variables - Pokemon height in meters, Pokemon defense
Dependent variable - Pokemon weight in kilograms
"""
height_defense = pokemon[["Height_m", "Defense"]].to_numpy()

# Perform multiple linear regression
multiple_linear = ml_lib.MultipleLinear(height_defense, weight)

# Output results
print("\nMultiple linear regression predicting Pokemon weight (kg) from height (m) and defense.")
print(f"Weight = {multiple_linear.beta[0]:.3f} + {multiple_linear.beta[1]:.3f} * height + {multiple_linear.beta[2]:.3f} * defense")
print(f"R^2    = {multiple_linear.R2_train:.3f}")
print(f"RMSE   = {multiple_linear.RMSE_train:.3f}")

# Set up the 3d plot
fig = plt.figure()
ax = fig.add_subplot(projection='3d')

# Plot the points
ax.scatter(height_defense[:, 0], height_defense[:, 1], weight, c="darkblue", s=2.0)

# Plot the plane given by multiple linear regression
x_min = np.min(height_defense[:, 0])
x_max = np.max(height_defense[:, 0])
y_min = np.min(height_defense[:, 1])
y_max = np.max(height_defense[:, 1])
x, y = np.meshgrid(np.linspace(x_min, x_max, num=10), np.linspace(y_min, y_max, num=10))
z = multiple_linear.beta[0] + multiple_linear.beta[1] * x + multiple_linear.beta[2] * y
ax.plot_surface(x, y, z, alpha=0.2)

# Label the axes and display the plot
ax.set_xlabel("Height (m)")
ax.set_ylabel("Defense")
ax.set_zlabel("Weight (kg)")
plt.show()


"""
Multiple linear regression on Pokemon data
Independent variables - Pokemon height in meters, Pokemon defense, attack, and HP
Dependent variable - Pokemon weight in kilograms
"""
poke_stats = pokemon[["Height_m", "Defense", "Attack", "HP"]].to_numpy()

# Perform multiple linear regression
multiple_linear2 = ml_lib.MultipleLinear(poke_stats, weight)

# Output results
print("\nMultiple linear regression predicting Pokemon weight (kg) from height (m), defense, attack, and HP.")
print("Weight = {0:.3f} + {1:.3f} * height + {2:.3f} * defense + {3:.3f} * attack + {4:.3f} * HP".format(*tuple(multiple_linear2.beta)))
print(f"R^2    = {multiple_linear2.R2_train:.3f}")
print(f"RMSE   = {multiple_linear2.RMSE_train:.3f}")
