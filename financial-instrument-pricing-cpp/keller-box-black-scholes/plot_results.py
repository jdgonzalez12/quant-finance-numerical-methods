import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams["axes.facecolor"] = "white"
plt.rcParams["figure.facecolor"] = "white"
plt.rcParams["axes.edgecolor"] = "black"
plt.rcParams["grid.color"] = "0.85"
plt.rcParams["grid.linestyle"] = "-"

df = pd.read_csv("keller_box_solution.csv")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

ax1.plot(df["S"], df["V_closed_form"], "k-", lw=2.5, label="Closed-form Black-Scholes")
ax1.plot(df["S"][::5], df["V_box"][::5], "o", color="tab:blue", ms=4, label="Keller Box (O(J) solve)")
ax1.set_xlabel("S")
ax1.set_ylabel("V(S, t=0)")
ax1.set_title("Keller Box option value vs. closed form")
ax1.legend()
ax1.grid(True, alpha=0.5)

ax2.semilogy(df["S"], df["abs_error"].clip(lower=1e-12), color="tab:red")
ax2.set_xlabel("S")
ax2.set_ylabel("|V_Box - V_closed_form| (log scale)")
ax2.set_title("Pointwise error across the full grid")
ax2.grid(True, which="both", alpha=0.5)

fig.tight_layout()
fig.savefig("keller_box_solution.png", dpi=150)
print("saved keller_box_solution.png")
