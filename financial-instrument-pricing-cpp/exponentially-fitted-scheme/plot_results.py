import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams["axes.facecolor"] = "white"
plt.rcParams["figure.facecolor"] = "white"
plt.rcParams["axes.edgecolor"] = "black"
plt.rcParams["grid.color"] = "0.85"
plt.rcParams["grid.linestyle"] = "-"

df = pd.read_csv("part_a_solution.csv")

fig, ax = plt.subplots(figsize=(8, 5.5))
ax.plot(df["x"], df["exact"], "k-", lw=3, label="Exact solution", zorder=1)
ax.plot(df["x"], df["centered"], "o-", color="tab:red", ms=4, lw=1,
        label="Standard centered scheme (Pe=2.5 > 1)")
ax.plot(df["x"], df["fitted"], "s--", color="tab:blue", ms=4, lw=1,
        label="Exponentially fitted scheme (Il'in)")
ax.set_xlabel("x")
ax.set_ylabel("u(x)")
ax.set_title("sigma*u'' + mu*u' = 0: centered scheme oscillates, fitted scheme is exact")
ax.legend()
ax.grid(True, alpha=0.5)

fig.tight_layout()
fig.savefig("exponentially_fitted_scheme_solution.png", dpi=150)
print("saved exponentially_fitted_scheme_solution.png")
