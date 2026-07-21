import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams["axes.facecolor"] = "white"
plt.rcParams["figure.facecolor"] = "white"
plt.rcParams["axes.edgecolor"] = "black"
plt.rcParams["grid.color"] = "0.85"
plt.rcParams["grid.linestyle"] = "-"

df = pd.read_csv("asian_option_value_surface.csv")

S_vals = np.sort(df["S"].unique())
A_vals = np.sort(df["A"].unique())
V = df.pivot(index="A", columns="S", values="V").reindex(index=A_vals, columns=S_vals).values

fig, ax = plt.subplots(figsize=(7.5, 6))
mesh = ax.pcolormesh(S_vals, A_vals, V, shading="auto", cmap="viridis")
fig.colorbar(mesh, ax=ax, label="V(S, A, t=0)")
ax.set_xlabel("S (underlying price)")
ax.set_ylabel("A (running average)")
ax.set_title("Arithmetic Asian call value surface (ADI, ~t=0)")

fig.tight_layout()
fig.savefig("asian_option_value_surface.png", dpi=150)
print("saved asian_option_value_surface.png")
