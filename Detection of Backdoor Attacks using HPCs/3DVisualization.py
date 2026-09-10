import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd

df = pd.read_csv("trace.csv")

fig = plt.figure(figsize = (8, 6))

ax = fig.add_subplot(111, projection = "3d")

ax.scatter(
    df["cache-references"],
    df["cycles"],
    df["LLC-loads"]
)

ax.set_xlabel("cache-references")
ax.set_ylabel("cycles")
ax.set_zlabel("LLC-loads")

plt.show()