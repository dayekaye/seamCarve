#!/usr/bin/env python
import sys

import numpy as np
from scipy.misc import imread, imsave

def remove_vertical_seam(image, seam):
    """Removes the given seam from the image. Seam should be a list of
       X-coordinates for the seam pixels in top-down order."""
    return np.array([np.concatenate([row[:s], row[s+1:]], axis=0) for row, s in zip(image, seam)])

def remove_minimal_seam(image, energy_function, seam_type="ver"):
    """Removes the minimal seam (according to the given energy function) from
    the image. Returns the image without the seam and the cost of the seam
    removal operation (the sum of the energy removed from the image)."""
    if seam_type == "hor":
        image = image.swapaxes(0,1)

    seam, cost = minimal_seam(compute_seam_costs(energy_function(image)))
    result = remove_vertical_seam(image, seam)

    if seam_type == "hor":
        result = result.swapaxes(0,1)

    return result, cost

def gradient_magnitude(image):
    """Returns the L1 gradient magnitude of the given image."""
    # Compute the horizontal (dx) and vertical (dy) differences in pixel
    # values.
    dx = np.roll(image, -1, axis = 1) - image
    dy = np.roll(image, -1, axis = 0) - image

    # Compute the L1 gradient magnitude (sum of absolute values of dx and dy).
    magnitude = (abs(dx) + abs(dy))

    # Return the average gradient magnitude across the image color channels.
    return magnitude.mean(-1)

def compute_seam_costs(energy):
    n, m = energy.shape

    # Create and fill in the matrix M with minimal seam cost entries according
    # to the dynamic programming rule:
    #     M(i, j) = e(i, j) + min( M(i-1, j-1), M(i-1, j), M(i-1, j+1) )
    M = np.zeros((n, m+2), np.float32)
    M[0, 1:-1] = energy[0]
    M[:, 0] = np.inf; M[:, -1] = np.inf

    for i in range(1, n):
        M[i, 1:-1] = energy[i] + np.min([M[i-1, :-2], M[i-1, 1:-1], M[i-1, 2:]], axis=0)
    M = M[:, 1:-1]

    return M

def minimal_seam(M):
    """Find the seam with minimal energy cost given the provided seam cost
    matrix M. Returns the X-coordinates of the minimal-cost vertical seam in
    top-down order."""
    path = []

    # Compute the bottom-up path of pixel X-coordinates for the seam with
    # minimal cost.
    M = np.pad(M, ((0,0), (1,1)), mode="constant", constant_values=(np.inf,))
    path.append(M[-1].argmin())
    cost = M[-1, path[0]]

    for i in range(2, M.shape[0]+1):
        prev = path[-1]
        path.append(prev + M[-i, prev-1:prev+2].argmin() - 1)

    path = np.array(path) - 1

    # Return the top-down seam X-coordinates and the total energy cost of
    # removing that seam.
    return np.asarray(path)[::-1], cost

def compute_ordering(image, target_size, energy_function):
    """Compute the optimal order of horizontal and vertical seam removals to
    achieve the given target image size. Order should be returned as a list of
    0 or 1 values corresponding to horizontal and vertical seams
    respectively."""
    r = image.shape[0] - target_size[0] + 1
    c = image.shape[1] - target_size[1] + 1
    if r < 0 or c < 0:
        raise ValueError("Target size must be smaller than the input size.")
    return [0,1] * min(r-1, c-1) + [0] * max(r-c, 0) + [1] * max(c-r, 0)

def resize(image, target_size, energy_function=gradient_magnitude):
    output = image.copy()
    order = compute_ordering(output, target_size, energy_function)

    for seam_type in order:
        if seam_type == 0:
            output = output.swapaxes(0,1)

        energy = energy_function(output)
        M = compute_seam_costs(energy)
        seam, cost = minimal_seam(M)
        output = remove_vertical_seam(output, seam)

        if seam_type == 0:
            output = output.swapaxes(0,1)

    assert(output.shape[0] == target_size[0] and
           output.shape[1] == target_size[1])
    return output



if __name__ == "__main__":
    try:
        in_fn, h, w, out_fn = sys.argv[1:]
        h, w = int(h), int(w)
    except:
        print("Usage: python p2.py FILE TARGET_HEIGHT TARGET_WIDTH OUTPUT")
        exit(1)

    image = imread(in_fn).astype(np.float32) / 255.
    resized = resize(image, (h,w))
    imsave(out_fn, resized)
