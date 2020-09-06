import numpy as np
from skimage import measure
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


volume = np.load("../arrays/Seed_00/eroded.npy")
verts, faces, normals, values = measure.marching_cubes(volume, 0.5, allow_degenerate=False)

fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')


ax.plot_trisurf(verts[:, 0], verts[:, 1], verts[:, 2], triangles=faces, cmap="terrain", antialiased=False)
plt.tight_layout()
plt.show()