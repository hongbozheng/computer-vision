#!/usr/bin/env python3


import argparse
import config
import matplotlib.pyplot as plt
import numpy as np
import os
import PIL.Image
import scipy.linalg
from ransac import *


def normalize(coords: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    N = coords.shape[0]
    mu = np.mean(a=coords, axis=0)

    sig_x = np.std(a=coords[:, 0])
    sig_y = np.std(a=coords[:, 1])
    # l2_norm_mu = numpy.mean(a=(coords-mu)**2, axis=0)
    # coeff = 1/numpy.sqrt(numpy.sum(a=l2_norm_mu)/2)

    T = np.array(object=[[np.sqrt(2)/sig_x, 0, -np.sqrt(2)/sig_x*mu[0]],
                            [0, np.sqrt(2)/sig_y, -np.sqrt(2)/sig_y*mu[1]],
                            [0, 0, 1]])
    # T = numpy.array(object=[[coeff, 0, -coeff*mu[0]],
    #                         [0, coeff, -coeff*mu[1]],
    #                         [0, 0, 1]])

    ones = np.ones(shape=(N, 1))
    coords = np.hstack(tup=(coords, ones), dtype=np.float64)
    coords = coords@T.T

    return coords[:, :2], T


def solve_mat_F(coords_0: np.ndarray, coords_1: np.ndarray) -> np.ndarray:
    A = []

    for i in range(coords_0.shape[0]):
        row = [coords_1[i][0] * coords_0[i][0], coords_1[i][0] * coords_0[i][1], coords_1[i][0],
               coords_1[i][1] * coords_0[i][0], coords_1[i][1] * coords_0[i][1], coords_1[i][1],
               coords_0[i][0], coords_0[i][1], 1]

        A.append(row)

    A = np.vstack(tup=A, dtype=np.float64)

    U, s, V = scipy.linalg.svd(a=A)
    F = V[-1].reshape(3, 3)

    U, s, V = scipy.linalg.svd(a=F)
    S = np.diag(v=s)
    S[-1] = 0
    F = U@S@V

    return F


def fit_fundamental(matches: np.ndarray, norm: bool) -> np.ndarray:
    coords_0 = matches[:, 0:2]
    coords_1 = matches[:, 2:4]

    if norm:
        coords_0, T0 = normalize(coords=coords_0)
        coords_1, T1 = normalize(coords=coords_1)


    F = solve_mat_F(coords_0=coords_0, coords_1=coords_1)

    if norm:
        F = T1.T@F@T0

    return F


def plt_matches(
        matches: np.ndarray,
        mat_0: np.ndarray,
        mat_1: np.ndarray,
) -> plt.Figure:
    # this is a N x 4 file where the first two numbers of each row
    # are coordinates of corners in the first image and the last two
    # are coordinates of corresponding corners in the second image:
    # matches(i,1:2) is a point in the first image
    # matches(i,3:4) is a corresponding point in the second image

    # display two images side-by-side with matches
    # this code is to help you visualize the matches, you don't need
    # to use it to produce the results for the assignment

    N = matches.shape[0]

    I3 = np.zeros(shape=(mat_0.shape[0], mat_0.shape[1] * 2, 3))
    I3[:, :mat_0.shape[1], :] = mat_0
    I3[:, mat_0.shape[1]:, :] = mat_1
    plt.rc(group="font", family="serif")
    fig, ax = plt.subplots()
    ax.set_aspect("equal")
    ax.imshow(np.array(I3).astype(dtype=np.uint8))
    ax.plot(matches[:, 0], matches[:, 1], '+r')
    ax.plot(matches[:, 2] + mat_0.shape[1], matches[:, 3], '+r')
    ax.plot(
        [matches[:, 0], matches[:, 2] + mat_0.shape[1]],
        [matches[:, 1], matches[:, 3]],
        color="red",
        linestyle="-.",
        alpha=0.35,
    )
    ax.set_title(f"{N} matches")

    plt.show()

    return fig


def calc_resid(
        F: np.ndarray,
        coords_0: np.ndarray,
        coords_1: np.ndarray,
) -> tuple[float, float]:
    N = coords_0.shape[0]

    ones = np.ones(shape=(N, 1), dtype=np.float64)
    coords_0 = np.hstack(tup=(coords_0, ones), dtype=np.float64)
    coords_1 = np.hstack(tup=(coords_1, ones), dtype=np.float64)

    resid = []
    for i in range(N):
        resid.append(abs(coords_1[i]@F@coords_0[i].T))

    avg_resid = np.mean(a=resid, dtype=np.float64)
    resid = np.sum(a=resid, dtype=np.float64)

    return avg_resid, resid


def plt_epipolar(
        matches: np.ndarray,
        F: np.ndarray,
        mat: np.ndarray,
        norm: bool,
        avg_resid: float
) -> plt.Figure:
    N = matches.shape[0]

    # display second image with epipolar lines reprojected
    # from the first image

    # first, fit fundamental matrix to the matches
    M = np.c_[matches[:, 0:2], np.ones(shape=(N, 1))].transpose()
    L1 = np.matmul(F, M).transpose()  # transform points from
    # the first image to get epipolar lines in the second image

    # find points on epipolar lines L closest to matches(:,3:4)
    l = np.sqrt(L1[:, 0] ** 2 + L1[:, 1] ** 2)
    L = np.divide(L1, np.kron(a=np.ones((3, 1)), b=l).transpose())
    pt_line_dist = np.multiply(
        L, np.c_[matches[:, 2:4], np.ones(shape=(N, 1))]).sum(axis=1)
    closest_pt = matches[:, 2:4] - np.multiply(
        L[:, 0:2], np.kron(a=np.ones(shape=(2, 1)), b=pt_line_dist).transpose())

    # find endpoints of segment on epipolar line (for display purposes)
    # offset from the closest point is 10 pixels
    pt1 = closest_pt - np.c_[L[:, 1], -L[:, 0]] * 10
    pt2 = closest_pt + np.c_[L[:, 1], -L[:, 0]] * 10

    # display points and segments of corresponding epipolar lines
    plt.rc(group="font", family="serif")
    fig, ax = plt.subplots()
    ax.set_aspect('equal')
    ax.imshow(np.array(object=mat).astype(dtype=np.uint8))
    ax.plot(matches[:, 2], matches[:, 3], '+r')
    ax.plot(
        [matches[:, 2], closest_pt[:, 0]],
        [matches[:, 3], closest_pt[:, 1]],
        'r',
    )
    ax.plot([pt1[:, 0], pt2[:, 0]], [pt1[:, 1], pt2[:, 1]], 'g')


    if norm:
        ax.set_title(f"Normalized     Residual: {avg_resid}")
    else:
        ax.set_title(f"Non-normalized     Residual: {avg_resid}")

    plt.show()

    return fig


def main():
    parser = argparse.ArgumentParser()
    allowed_files = config.filepaths.keys()
    parser.add_argument(
        "--input_file",
        "-f",
        type=str,
        choices=allowed_files,
        required=True,
        help="Input file",
    )
    parser.add_argument(
        "--normalized",
        "-n",
        type=bool,
        default=False,
        help="Normalized flag",
    )

    args = parser.parse_args()
    dir = args.input_file
    filepaths = config.filepaths[dir]
    img_filepath_0 = filepaths[0]
    img_filepath_1 = filepaths[1]
    norm = args.normalized

    # load images and match files for the first example
    img_0 = PIL.Image.open(fp=img_filepath_0)
    mat_0 = np.asarray(a=img_0, dtype=np.uint8)
    img_1 = PIL.Image.open(fp=img_filepath_1)
    mat_1 = np.asarray(a=img_1, dtype=np.uint8)

    if dir == "lab" or dir == "lib":
        if dir == "lab":
            logger.log_info("Processing lab images")
        else:
            dir = "library"
            logger.log_info("Processing library images")
        logger.log_info(f"Normalization: {norm}")

        matches_filepath = filepaths[2]
        matches = np.loadtxt(fname=matches_filepath)

        F = fit_fundamental(matches=matches, norm=norm)

        avg_resid, resid = calc_resid(
            F=F, coords_0=matches[:, :2], coords_1=matches[:, 2:])
        logger.log_info("Average Residual %f" % avg_resid)
        logger.log_info("Total Residual %f" % resid)

        fig_matches = plt_matches(matches=matches, mat_0=mat_0, mat_1=mat_1)

        fig = plt_epipolar(
            matches=matches, F=F, norm=norm, mat=mat_1, avg_resid=avg_resid)

        if config.imwrite:
            res_dir = os.path.join(config.res_dir, dir)
            if not os.path.exists(path=res_dir):
                os.makedirs(name=res_dir, exist_ok=True)

            fig_matches.savefig(
                fname=os.path.join(res_dir, "matches.png"),
                dpi=1000,
                format="png",
                bbox_inches="tight",
            )
            if norm:
                fig.savefig(
                    fname=os.path.join(res_dir, "epipolar_norm.png"),
                    dpi=1000,
                    format="png",
                    bbox_inches="tight",
                )
            else:
                fig.savefig(
                    fname=os.path.join(res_dir, "epipolar.png"),
                    dpi=1000,
                    format="png",
                    bbox_inches="tight",
                )
    elif dir == "gaudi" or dir == "house":
        logger.log_info(f"Processing {dir} images")
        norm = True
        logger.log_info(f"Normalization: {norm}")

        coords_0, coords_1 = comp_matches(mat_0=mat_0, mat_1=mat_1)

        logger.log_info("Performing ransac")
        H, inliners, avg_res = ransac(
            coords_0=coords_0,
            coords_1=coords_1,
            num_iters=config.ransac_num_iters,
            thres=config.ransac_thres,
        )

        fig_matches = plt_inlier_matches(
            mat_0=mat_0, mat_1=mat_1, inliers=inliners, avg_res=avg_res)

        F = fit_fundamental(matches=inliners, norm=True)

        avg_resid, resid = calc_resid(
            F=F, coords_0=inliners[:, :2], coords_1=inliners[:, 2:])
        logger.log_info("Average Residual %f" % avg_resid)
        logger.log_info("Total Residual %f" % resid)

        fig = plt_epipolar(
            matches=inliners, norm=norm, F=F, mat=mat_1, avg_resid=avg_resid)

    if config.imwrite:
        res_dir = os.path.join(config.res_dir, dir)
        if not os.path.exists(path=res_dir):
            os.makedirs(name=res_dir, exist_ok=True)

        fig_matches.savefig(
            fname=os.path.join(res_dir, "matches.png"),
            dpi=1000,
            format="png",
            bbox_inches="tight",
        )
        if norm:
            fig.savefig(
                fname=os.path.join(res_dir, "epipolar_norm.png"),
                dpi=1000,
                format="png",
                bbox_inches="tight",
            )
        else:
            fig.savefig(
                fname=os.path.join(res_dir, "epipolar.png"),
                dpi=1000,
                format="png",
                bbox_inches="tight",
            )

    return


if __name__ == "__main__":
    main()
