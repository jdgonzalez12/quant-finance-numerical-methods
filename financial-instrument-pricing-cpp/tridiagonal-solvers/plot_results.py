import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams["axes.facecolor"] = "white"
plt.rcParams["figure.facecolor"] = "white"
plt.rcParams["axes.edgecolor"] = "black"
plt.rcParams["grid.color"] = "0.85"
plt.rcParams["grid.linestyle"] = "-"

df = pd.read_csv("test1_solution.csv")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

ax1.plot(df["x"], df["exact"], "k-", lw=2.5, label="Exact: sin(x)/sin(1)")
ax1.plot(df["x"], df["LU"], "o", color="tab:blue", ms=4, label="Specialized LU")
ax1.plot(df["x"], df["DoubleSweep"], "x", color="tab:red", ms=5, label="Godunov double sweep")
ax1.set_xlabel("x")
ax1.set_ylabel("u(x)")
ax1.set_title("Test 1: u'' + u = 0, u(0)=0, u(1)=1")
ax1.legend()
ax1.grid(True, alpha=0.5)

err_lu = (df["LU"] - df["exact"]).abs()
err_ds = (df["DoubleSweep"] - df["exact"]).abs()
ax2.semilogy(df["x"], err_lu, "o-", color="tab:blue", ms=4, label="|LU - exact|")
ax2.semilogy(df["x"], err_ds, "x--", color="tab:red", ms=5, label="|DoubleSweep - exact|")
ax2.set_xlabel("x")
ax2.set_ylabel("absolute error (log scale)")
ax2.set_title("Both O(J) solvers agree with the closed form")
ax2.legend()
ax2.grid(True, which="both", alpha=0.5)

fig.tight_layout()
fig.savefig("tridiagonal_solvers_solution.png", dpi=150)
print("saved tridiagonal_solvers_solution.png")
