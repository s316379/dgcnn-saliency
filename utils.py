import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import proj3d

def render_cloud(cloud:np.ndarray, filename:str=None) -> None:
    x = cloud[:, 0]
    z = cloud[:, 1]
    y = cloud[:, 2]

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(x, y, z)
    ax.axis('off')
    plt.show()