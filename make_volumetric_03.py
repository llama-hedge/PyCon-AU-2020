import numpy as np
from config import HEIGHT, WIDTH, LENGTH, UNPADDED_HEIGHT, PAD
from itertools import product


heightmap = np.load("arrays/withmountain.npy")

# normalise for height
heightmap *= UNPADDED_HEIGHT / np.max(heightmap)

volume = np.zeros((LENGTH, WIDTH, HEIGHT))
volume[:, :, 0:PAD] = 1

for row, col in product(range(LENGTH), range(WIDTH)):
    height = int(heightmap[row, col])
    volume[row, col, PAD:height+PAD] = 1

np.save("arrays/volume", volume)
