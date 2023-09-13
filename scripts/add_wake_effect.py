# -*- coding: utf-8 -*-
# SPDX-FileCopyrightText: : 2017-2022 The PyPSA-Eur Authors
#
# SPDX-License-Identifier: MIT
# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pypsa

# %%
density = 7  # MW/km2
wind_farm_sizes = np.array([2, 10, 12])  # MW
capacity_factors_wake = np.array([54.1, 52.4, 51.9])  # yearly average capacity factors
capacity_factors_no_wake = np.array(
    [59.7, 59.7, 59.7]
)  # yearly average capacity factors
capacity_factors_modified = np.linalg.solve(
    np.array([[2, 0, 0], [2, 10, 0], [2, 10, 12]]),
    wind_farm_sizes.cumsum() * capacity_factors_wake,
)

wind_farm_coverage = wind_farm_sizes / 24  # km2
wake_losses_modified = 1 - (capacity_factors_modified / capacity_factors_no_wake)


# %%
def calculate_wake_turbine(CT, d0, k, x, HH=0, z=0, y=0):
    """
    Calculates the velocity deficit (U0-U1/U0) due to wake effect of a wind turbine based on the thrust coefficient (CT),
    rotor diameter (d0), wake decay rate (k), hub height (HH), and the distance from the turbine (x, z, y). Ref: https://doi.org/10.1016/j.renene.2014.01.002

    Args:
        CT (float): Thrust coefficient of the wind turbine. (different values, normally 0.8 or 0.75)
        d0 (float): Rotor diameter of the wind turbine.
        k (float): Wake recovery rate. For onshore 0.075, for offshore 0.04.
        HH (float): Hub height of the wind turbine.
        x (float): Distance from the turbine in the x-direction.
        z (float): Distance from the turbine in the z-direction.
        y (float): Distance from the turbine in the y-direction.

    Returns:
        float: The wake effect of the wind turbine.
    """
    beta = calculate_beta(CT)
    deficit = (
        1 - np.sqrt(1 - CT / (8 * (k * x / d0 + 0.2 * np.sqrt(beta)) ** 2))
    ) * np.exp(
        -1
        / (2 * (k * x / d0 + 0.2 * np.sqrt(beta)) ** 2)
        * (((z - HH) / d0) ** 2 + (y / d0) ** 2)
    )
    return deficit


# %%
def calculate_spacing_distance(alpha, spacing):
    """
    Calculates the spacing distance between turbines based on the angle of the
    wind direction.

    Parameters:
    alpha (float or list): Angle of the wind direction in degrees.
    spacing (float): Spacing distance between turbines.

    Returns:
    float: The calculated wind angle adjusted spacing distance between turbines.
    """
    if isinstance(alpha, float):
        alpha = np.array([alpha])
    elif isinstance(alpha, list):
        alpha = np.array(alpha)
    alpha = [angle if angle <= 45 else 45 - angle % 45 for angle in alpha % 90]
    spacing = spacing / np.cos(np.deg2rad(alpha))
    return spacing.squeeze()


# %%
def calculate_wake_other(CT, d0, k, x):
    """
    Other model from https://doi.org/10.1002/we.1863

    Args:
        CT (float): Thrust coefficient of the wind turbine. (different values, normally 0.8 or 0.75)
        d0 (float): Rotor diameter of the wind turbine.
        k (float): Wake recovery rate. For onshore 0.075, for offshore 0.04.
        x (float): Distance from the turbine downstream.

    Returns:
        float: The wake effect of the wind turbine.
    """
    deficit = (1 - np.sqrt(1 - CT)) / (1 + k * x / d0) ** 2
    return deficit


# %%
def calculate_wake_farm(deficit, n):
    """
    Calculates the wake effect of a wind farm based on the velocity deficit (U0-U1/U0) of the wind turbines and the number of turbines in the wind farm in one row. From https://doi.org/10.1002/we.1863

    Args:
        deficit (float): Velocity deficit of the wind turbines.
        n (int): Number of turbines in the wind farm.

    Returns:
        float: The wake effect of the wind farm.
    """
    farm_deficit = (deficit**2 * n) ** 0.5
    return farm_deficit


# %%
def calculate_wake_turbine_i(i, U0, CT, d0, k):
    """ """
    wake = U0 * (1 - np.sqrt(1 - (CT * calculate_wake_farm_i(i, CT, d0, k) / U0) ** 2))

    return farm_deficit


# %%
def calculate_beta(CT):
    return 1 / 2 * (1 + np.sqrt(1 - CT)) / np.sqrt(1 - CT)


# %%
def calculate_epsilon(CT):
    return 0.2 * np.sqrt(calculate_beta(CT))


# %%
def calculate_sigma(k, x, d0):
    """
    Calculates the wake standard deviation (sigma) based on the wake recovery
    rate (k), the distance from the turbine (x), and the rotor diameter (d0).

    Args:
    - k (float): wake decay coefficient
    - x (float): distance from the turbine
    - d0 (float): rotor diameter

    Returns:
    - sigma (float): wake standard deviation
    """
    # wake half-width
    epsilon = calculate_epsilon(CT)
    return k * x + epsilon / d0


# %%
def calculate_epsilon(CT):
    return 0.2 * calculate_beta(CT)


# %%
calculate_wake_turbine(CT, d0, k, HH, x, z, y)
# %%
d0 = 80
x = 8 * d0
HH = 100
z = HH
y = 0
CT = 0.8
k = 0.04
xrange = np.arange(9)
fig, axes = plt.subplots(3, 3)
i = 3
z = np.arange(0, 2 * d0)
for ax in iter(axes.flatten()):
    x = d0 * (i)
    ax.plot(calculate_wake_turbine(CT, d0, k, HH, x, z, y), z / d0)
    ax.set_ylabel("z/d")
    ax.set_xlabel("$\Delta U/U_0$")
    ax.set_title(f"x/d={i}")
    ax.set_xlim(0, 0.5)
    i += 1
# %%
d0 = 80
x = np.arange(0, 16 * d0)
HH = 100
z = HH
y = 0
CT = 0.8
k = 0.04

fig, ax = plt.subplots(2, 1)
ax[0].plot(x / d0, calculate_wake_turbine(CT, d0, k, HH, x, z, y))
ax[0].set_ylabel("$\Delta U/U_0$")
ax[0].set_xlabel("x/d")
ax[0].grid()
ax[1].plot(x / d0, 1 - calculate_wake_turbine(CT, d0, k, HH, x, z, y))
ax[1].set_ylabel("$U/U_0$")
ax[1].set_xlabel("x/d")
ax[1].grid()
# %%
# model in for wind farm with spacing in main wind direction

alpha = np.arange(0, 361, 15)
spacing = 8 * d0
d0 = 80
CT = 0.8
k = 0.04
spacing = calculate_spacing_distance(alpha, spacing)
fig, ax = plt.subplots()
velocity_deficit = calculate_wake_turbine(CT, d0, k, x=spacing)

ax.plot(alpha, calculate_wake_turbine(CT, d0, k, x=spacing))


# %%
# other model example
d0 = 80
x = np.arange(0, 20 * d0)
CT = 0.8
k = 0.04

fig, ax = plt.subplots()
ax.plot(x / d0, calculate_wake_other(CT, d0, k, x))
ax.set_ylabel("$\Delta U/U_0$")
ax.set_xlabel("x/d")
# %%
