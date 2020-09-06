import numpy as np
import random
from skimage.morphology import skeletonize
from skimage.transform import swirl
from skimage.filters import gaussian
from config import SEED, WIDTH, LENGTH, PAD
from simplex_base_01 import noise_layer


if __name__ == "__main__":
    random.seed(SEED)
    base_array = np.load("arrays/opensimplexbase.npy")

    # create a mask identifying which is island based on height
    waterline = np.percentile(base_array, 70)
    land = base_array > waterline

    x_perturb = random.randint(-int(0.05 * WIDTH), int(0.05 * WIDTH))
    y_perturb = random.randint(-int(0.05 * LENGTH), int(0.05 * LENGTH))
    offset_land = np.roll(land, (x_perturb, y_perturb), axis=(1, 0))
    offset_land[land == 1] = 0

    line = skeletonize(offset_land)
    line = swirl(line, strength=0.5*random.random(), radius=WIDTH*LENGTH/4,
                 center=(random.randint(PAD, WIDTH - PAD - 1), random.randint(PAD, LENGTH - PAD - 1)))
    line = gaussian(line, sigma=1.5)
    line *= noise_layer(32, length=LENGTH, width=WIDTH) + 1

    base_array += line
    np.save("arrays/withmountain", base_array)