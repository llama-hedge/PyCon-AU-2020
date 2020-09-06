import numpy as np
import random
from config import SEED, WIDTH, LENGTH, HEIGHT
import time

vectors = [[i, j, k] for i in range(-1, 2) for j in range(-1, 2) for k in range(-1, 2) if (i, j, k) != (0, 0, 0)]
unit_vectors = [v / np.linalg.norm(v) for v in vectors]

coord_cube_flat = np.array([(i, j, k) for i in range(-1, 2) for j in range(-1, 2) for k in range(-1, 2)], dtype=int)
coord_cube = coord_cube_flat.reshape((3, 3, 3, 3))


class Droplet:
    def __init__(self, grid):
        self.grid = grid
        # timer represents evaporation
        self.timer = 100
        self.contents = 0
        self.pos = np.array([random.randint(0, WIDTH-2), random.randint(0, LENGTH-2), HEIGHT-2], dtype=int)
        self.velocity = np.array([0, 0, 0])
        self.valid = True

    @property
    def ahead_vector(self):
        # returns vector closest to velocity
        return choose_direction(self.velocity)

    @property
    def ahead_pos(self):
        return np.array([self.pos[i]+self.ahead_vector[i] for i in range(3)], dtype=int)

    @property
    def b_pos(self):
        # need this for gravity

        return np.array([self.pos[0], self.pos[1], self.pos[2]-1], dtype=int)

    @property
    def volume(self):
        return self.timer // 20

    @property
    def capacity(self):
        return np.linalg.norm(self.velocity) * self.volume // 10

    @property
    def nhd(self):
        # neighbourhood of voxels adjacent to droplet
        return self.grid[self.pos[0]-1:self.pos[0]+2, self.pos[1]-1:self.pos[1]+2, self.pos[2]-1:self.pos[2]+2]

    def is_inside(self):
        if 0 < self.pos[0] < LENGTH -1 and 0 < self.pos[1] < WIDTH - 1 and 0 < self.pos[2] < HEIGHT - 1:
            return True
        return False

    def erode_and_deposit(self):
        if self.contents < self.capacity:

            if self.nhd[1, 1, 1] > 0:
                # erode from occupied voxel if possible
                self.contents += 1
                self.grid[tuple(self.pos)] -= 0.2
            elif self.nhd[1, 1, 0] > 0:
                # if not, erode from voxel immediately ahead
                self.contents += 1
                self.grid[self.pos[0], self.pos[1], self.pos[2] -1] -= 0.2
            elif self.nhd[tuple(1 + self.ahead_vector[i] for i in range(3))] > 0:
                # if that's not possible either, erode from voxel immediately below
                self.contents += 1
                self.grid[tuple(self.ahead_pos)] -= 0.2

        while self.contents > self.capacity and self.grid[tuple(self.b_pos)] > 0.5:
            # deposit into voxel immediately below if possible, otherwise deposit into occupied voxel if possible
            # only deposit if the droplet is on the ground
            if self.nhd[1, 1, 0] < 1:
                self.contents -= 1
                self.grid[tuple(self.b_pos)] += 0.2
            elif self.nhd[1, 1, 1] < 1:
                self.contents -= 1
                self.grid[tuple(self.pos)] += 0.2
            else:
                break

    def move(self):
        if not self.is_inside():
            # cancel the move
            return
        # if there is space below, increment downward vertical velocity by 1
        if self.grid[tuple(self.b_pos)] < 0.5 and self.velocity[2] > -10:
            self.velocity[2] -= 1
        # move one square (including diagonals) in the closest to the direction of velocity as limited by the grid
        direction = choose_direction(self.velocity)
        # check that voxel in that direction is not occupied
        if self.nhd[tuple(1+i for i in direction)] < 0.5:
            self.pos += direction
            if not self.is_inside():        # has been changed
                self.valid = False
                return
        else:
            # first find a direction adjacent to the current velocity
            # possible movements are sorted according to how close they are to the current velocity
            options = sorted(vectors, key=lambda x: sum(abs((x[i] - direction[i])**2) for i in range(3)))
            for adjacent_vector in options:
                if self.nhd[tuple(i+1 for i in adjacent_vector)] < 0.5:
                    self.pos += adjacent_vector
                    if not self.is_inside():
                        self.valid = False
                        return
                    # add random variation to minimise diagonal artifacts
                    self.velocity = self.velocity + np.array(adjacent_vector, dtype=float) + 2 * (np.array((random.random(), random.random(), random.random())) - 0.5)
                    break

        # evaporate
        self.timer -= 1
        if self.timer <= 0:
            self.valid = False

    def check_stuck(self):
        # change is so that validity can be changed in lots of places in a way that makes sense.
        if np.linalg.norm(self.velocity) < 1:
            self.valid = False


def choose_direction(v):
    # choose vector closest to velocity
    if all(v == 0):
        return [0, 0, 0]
    v = v / np.linalg.norm(v)
    best_diff = np.inf
    for i, u in enumerate(unit_vectors):
        diff = np.linalg.norm(u - v)
        if diff < best_diff:
            best_diff = diff
            best_index = i
    return vectors[best_index]


if __name__ == "__main__":
    random.seed(SEED)
    volume = np.load("arrays/volume.npy")
    start = time.time()
    for i in range(25000):
        droplet = Droplet(volume)
        while droplet.valid:
            droplet.erode_and_deposit()
            droplet.move()
            droplet.check_stuck()
        if i % 1000 == 0:
            print("progress:", i)
    end = time.time()
    print(end - start)
    np.save("arrays/eroded.npy", volume)
