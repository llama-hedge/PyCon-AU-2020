from opensimplex import OpenSimplex
from config import SEED, WIDTH, LENGTH, PAD, UNPADDED_LENGTH, UNPADDED_WIDTH
import numpy as np


xs = np.linspace(start=0, stop=UNPADDED_WIDTH - 1, num=UNPADDED_WIDTH)
ys = np.linspace(start=0, stop=UNPADDED_LENGTH - 1, num=UNPADDED_LENGTH)
xs, ys = np.meshgrid(xs, ys)

xs_longer = np.linspace(start=0, stop=WIDTH - 1, num=WIDTH)
ys_longer = np.linspace(start=0, stop=LENGTH - 1, num=LENGTH)
xs_longer, ys_longer = np.meshgrid(xs_longer, ys_longer)

gen = OpenSimplex(seed=SEED)


def gradient_array():
    x_rad = np.ones_like(xs) - abs(xs - UNPADDED_WIDTH / 2) / (UNPADDED_WIDTH*0.5)
    y_rad = np.ones_like(ys) - abs(ys - UNPADDED_WIDTH / 2) / (UNPADDED_WIDTH*0.5)
    return np.sqrt(x_rad * y_rad)


def noise_layer(frequency, length=UNPADDED_LENGTH, width=UNPADDED_WIDTH):
    layer = np.zeros((length, width))
    for x in range(length):
        for y in range(width):
            layer[x, y] += gen.noise2d(x * frequency / length, y * frequency / width)
    return layer


if __name__ == "__main__":

    simplex_array = np.zeros((UNPADDED_LENGTH, UNPADDED_WIDTH))

    for i in range(2, 9):
        simplex_array += noise_layer(2**i) * (2 ** -i)

    grad = gradient_array()
    simplex_array -= np.min(simplex_array)
    simplex_array *= grad

    simplex_array = np.pad(simplex_array, pad_width=PAD, mode="constant")
    simplex_array[simplex_array < 0] = 0

    np.save("arrays/opensimplexbase", simplex_array)