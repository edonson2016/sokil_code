import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd

data_obj = pd.read_csv("test/6test.csv")
x = data_obj["x1"].to_numpy()
y = data_obj["y1"].to_numpy()
z = data_obj["z1"].to_numpy()

fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x, y, z, s=1, c=z, cmap='viridis') # Color by Z-value
ax.set_xlabel('X Coordinate')
ax.set_ylabel('Y Coordinate')
ax.set_zlabel('Z Coordinate')
ax.set_title('LiDAR Point Cloud (3D View)')
plt.show()