#!/usr/bin/env python3


import argparse
import config
import matplotlib.pyplot
import numpy
import os
import PIL.Image
import scipy.linalg


def normalize(coords: numpy.ndarray) -> tuple[numpy.ndarray, numpy.ndarray]:
    mu = numpy.mean(a=coords, axis=0)

    sig_x = numpy.std(a=coords[:, 0])
    sig_y = numpy.std(a=coords[:, 1])

    T = numpy.array(object=[[numpy.sqrt(2)/sig_x, 0, -numpy.sqrt(2)/sig_x*mu[0]],
                            [0, numpy.sqrt(2)/sig_y, -numpy.sqrt(2)/sig_y*mu[1]],
                            [0, 0, 1]])

    ones = numpy.ones(shape=(coords.shape[0], 1))
    coords = numpy.concatenate((coords, ones), axis=1)
    coords = (T@coords.T).T

    return coords[:, :2], T


def solve_mat_F(coords_0: numpy.ndarray, coords_1: numpy.ndarray) -> numpy.ndarray:
    A = []

    for i in range(coords_0.shape[0]):
        row = [coords_1[i][0] * coords_0[i][0], coords_1[i][0] * coords_0[i][1], coords_1[i][0],
               coords_1[i][1] * coords_0[i][0], coords_1[i][1] * coords_0[i][1], coords_1[i][1],
               coords_0[i][0], coords_0[i][1], 1]

        A.append(row)

    A = numpy.vstack(tup=A, dtype=numpy.float64)

    U, s, V = scipy.linalg.svd(a=A)
    F = V[-1].reshape(3, 3)
    F /= F[2, 2]

    U, s, V = scipy.linalg.svd(a=F)
    S = numpy.diag(v=s)
    S[-1] = 0
    F = U@S@V

    return F


def fit_fundamental(matches: numpy.ndarray, norm: bool) -> numpy.ndarray:
    coords_0 = matches[:, 0:2]
    coords_1 = matches[:, 2:4]

    if norm:
        coords_0, T0 = normalize(coords=coords_0)
        coords_1, T1 = normalize(coords=coords_1)

    # sampled_indices = numpy.random.choice(a=coords_0.shape[0], size=8, replace=False)
    # sampled_coords_0 = coords_0[sampled_indices]
    # sampled_coords_1 = coords_1[sampled_indices]

    F = solve_mat_F(coords_0=coords_0, coords_1=coords_1)

    if norm:
        F = T1.T@F@T0

    return F


def plt_matches(matches: numpy.ndarray, mat_0: numpy.ndarray, mat_1: numpy.ndarray) -> matplotlib.pyplot.Figure:
    # this is a N x 4 file where the first two numbers of each row
    # are coordinates of corners in the first image and the last two
    # are coordinates of corresponding corners in the second image:
    # matches(i,1:2) is a point in the first image
    # matches(i,3:4) is a corresponding point in the second image

    # display two images side-by-side with matches
    # this code is to help you visualize the matches, you don't need
    # to use it to produce the results for the assignment

    N = matches.shape[0]

    I3 = numpy.zeros(shape=(mat_0.shape[0], mat_0.shape[1] * 2, 3))
    I3[:, :mat_0.shape[1], :] = mat_0
    I3[:, mat_0.shape[1]:, :] = mat_1
    matplotlib.pyplot.rc(group="font", family="serif")
    fig, ax = matplotlib.pyplot.subplots()
    ax.set_aspect("equal")
    ax.imshow(numpy.array(I3).astype(dtype=numpy.uint8))
    ax.plot(matches[:, 0], matches[:, 1], '+r')
    ax.plot(matches[:, 2] + mat_0.shape[1], matches[:, 3], '+r')
    ax.plot([matches[:, 0], matches[:, 2] + mat_0.shape[1]], [matches[:, 1], matches[:, 3]], color="red", linestyle="-.",
            alpha=0.35)
    ax.set_title(f"{N} matches")

    matplotlib.pyplot.show()

    return fig


def calc_resid(F: numpy.ndarray, coords_0: numpy.ndarray, coords_1: numpy.ndarray) -> float:
    ones = numpy.ones(shape=(coords_0.shape[0], 1), dtype=numpy.float64)
    coords_0 = numpy.hstack(tup=(coords_0, ones), dtype=numpy.float64)
    coords_1 = numpy.hstack(tup=(coords_1, ones), dtype=numpy.float64)
    print(coords_0.shape)
    coords_0_xf = coords_0@F
    print(coords_0_xf.shape)
    l2_norm = scipy.linalg.norm(a=coords_0_xf, axis=1)
    print(l2_norm.shape)
    coords_0_xf = coords_0_xf/l2_norm[:, numpy.newaxis]
    print(coords_0_xf.shape)
    dist = numpy.sum(a=numpy.multiply(coords_0_xf, coords_1), axis=1)
    resid = numpy.mean(a=numpy.square(dist))
    print(dist)
    return resid


def plt_epipolar(matches: numpy.ndarray, norm: bool, F: numpy.ndarray, mat: numpy.ndarray) -> matplotlib.pyplot.Figure:
    N = matches.shape[0]

    # display second image with epipolar lines reprojected
    # from the first image

    # first, fit fundamental matrix to the matches
    M = numpy.c_[matches[:, 0:2], numpy.ones(shape=(N, 1))].transpose()
    L1 = numpy.matmul(F, M).transpose()  # transform points from
    # the first image to get epipolar lines in the second image

    # find points on epipolar lines L closest to matches(:,3:4)
    l = numpy.sqrt(L1[:, 0] ** 2 + L1[:, 1] ** 2)
    L = numpy.divide(L1, numpy.kron(a=numpy.ones((3, 1)), b=l).transpose())  # rescale the line
    pt_line_dist = numpy.multiply(L, numpy.c_[matches[:, 2:4], numpy.ones(shape=(N, 1))]).sum(axis=1)
    closest_pt = matches[:, 2:4] - numpy.multiply(L[:, 0:2],
                                                  numpy.kron(a=numpy.ones(shape=(2, 1)), b=pt_line_dist).transpose())

    # find endpoints of segment on epipolar line (for display purposes)
    pt1 = closest_pt - numpy.c_[L[:, 1], -L[:, 0]] * 10  # offset from the closest point is 10 pixels
    pt2 = closest_pt + numpy.c_[L[:, 1], -L[:, 0]] * 10

    # display points and segments of corresponding epipolar lines
    matplotlib.pyplot.rc(group="font", family="serif")
    fig, ax = matplotlib.pyplot.subplots()
    ax.set_aspect('equal')
    ax.imshow(numpy.array(object=mat).astype(dtype=numpy.uint8))
    ax.plot(matches[:, 2], matches[:, 3], '+r')
    ax.plot([matches[:, 2], closest_pt[:, 0]], [matches[:, 3], closest_pt[:, 1]], 'r')
    ax.plot([pt1[:, 0], pt2[:, 0]], [pt1[:, 1], pt2[:, 1]], 'g')

    resid = calc_resid(F, coords_0=matches[:, 0:2], coords_1=matches[:, 2:])
    if norm:
        ax.set_title(f"Normalized     Residual: {resid}")
    else:
        ax.set_title(f"Unnormalized     Residual: {resid}")

    matplotlib.pyplot.show()

    return fig


def main():
    parser = argparse.ArgumentParser()
    allowed_files = config.filepaths.keys()
    parser.add_argument("--input_file", "-f", type=str, choices=allowed_files, required=True,
                        help="Input file")
    parser.add_argument("--normalized", "-n", type=bool, default=False, help="Normalized flag")

    args = parser.parse_args()
    dir = args.input_file
    filepaths = config.filepaths[dir]
    img_filepath_0 = filepaths[0]
    img_filepath_1 = filepaths[1]
    matches_filepath = filepaths[2]
    norm = args.normalized

    # load images and match files for the first example
    img_0 = PIL.Image.open(fp=img_filepath_0)
    mat_0 = numpy.asarray(a=img_0, dtype=numpy.uint8)
    img_1 = PIL.Image.open(fp=img_filepath_1)
    mat_1 = numpy.asarray(a=img_1, dtype=numpy.uint8)
    matches = numpy.loadtxt(fname=matches_filepath)

    fig = plt_matches(matches=matches, mat_0=mat_0, mat_1=mat_1)

    res_dir = os.path.join(config.res_dir, dir)
    if not os.path.exists(path=res_dir):
        os.makedirs(name=res_dir, exist_ok=True)

    if config.imwrite:
        fig.savefig(fname=os.path.join(res_dir, "matches.jpg"), dpi=1000, format="jpg", bbox_inches="tight")

    F = fit_fundamental(matches=matches, norm=norm)

    fig = plt_epipolar(matches=matches, norm=norm, F=F, mat=mat_1)

    if config.imwrite:
        if norm:
            fig.savefig(fname=os.path.join(res_dir, "epipolar_norm.jpg"), dpi=1000, format="jpg", bbox_inches="tight")
        else:
            fig.savefig(fname=os.path.join(res_dir, "epipolar.jpg"), dpi=1000, format="jpg", bbox_inches="tight")

    return


if __name__ == "__main__":
    main()