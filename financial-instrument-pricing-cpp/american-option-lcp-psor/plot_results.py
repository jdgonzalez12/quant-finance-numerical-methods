import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams["axes.facecolor"] = "white"
plt.rcParams["figure.facecolor"] = "white"
plt.rcParams["axes.edgecolor"] = "black"
plt.rcParams["grid.color"] = "0.85"
plt.rcParams["grid.linestyle"] = "-"

df = pd.read_csv("american_put_value.csv")

fig, ax = plt.subplots(figsize=(8, 5.5))
ax.plot(df["S"], df["V_american"], color="tab:blue", lw=2, label="American put (PSOR)")
ax.plot(df["S"], df["V_american_bs"], "--", color="tab:orange", lw=1.5,
        label="American put (Brennan-Schwartz)")
ax.plot(df["S"], df["V_european_intrinsic"], "k:", lw=1.5, label="Intrinsic value max(K-S,0)")
ax.set_xlabel("S")
ax.set_ylabel("Option value")
ax.set_title("American put value vs. underlying: PSOR and Brennan-Schwartz agree")
ax.legend()
ax.grid(True, alpha=0.5)

fig.tight_layout()
fig.savefig("american_put_value.png", dpi=150)
print("saved american_put_value.png")
