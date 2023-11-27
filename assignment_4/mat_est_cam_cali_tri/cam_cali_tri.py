#!/usr/bin/env python3


import argparse
import config
import logger
import matplotlib.pyplot
import numpy
import os
import scipy.linalg


def cam_cali(pts_3d: numpy.ndarray, pts_2d: numpy.ndarray) -> numpy.ndarray:
    """
    https://sites.cc.gatech.edu/classes/AY2016/cs4476_fall/results/proj3/html/ckobus3/index.html
    :param pts_3d: 3D coordinates
    :param pts_2d: 2D coordinates
    :return: projection matrix
    """
    N = pts_3d.shape[0]

    zeros = numpy.zeros(shape=4, dtype=numpy.float64)
    ones = numpy.ones(shape=(N, 1), dtype=numpy.float64)
    pts_homo = numpy.hstack(tup=(pts_3d, ones), dtype=numpy.float64)

    A = []

    for i in range(N):
        row_1 = numpy.hstack(tup=(pts_homo[i], zeros, -1*pts_2d[i, 0]*pts_homo[i]), dtype=numpy.float64)
        row_2 = numpy.hstack(tup=(zeros, pts_homo[i], -1*pts_2d[i, 1]*pts_homo[i]), dtype=numpy.float64)
        A.append(row_1)
        A.append(row_2)

    A = numpy.vstack(tup=A, dtype=numpy.float64)

    _, _, V = scipy.linalg.svd(a=A)
    M = V[-1].reshape(3, 4)

    return M


def evaluate_points(M: numpy.ndarray, pts_3d: numpy.ndarray, pts_2d: numpy.ndarray) -> tuple[numpy.ndarray, float]:
    """
    Visualize the actual 2D points and the projected 2D points calculated from
    the projection matrix
    You do not need to modify anything in this function, although you can if you
    want to
    :param M: projection matrix 3 x 4
    :param pts_3d: 3D points N x 3
    :param pts_2d: 2D points N x 2
    :return:
    """
    N = pts_3d.shape[0]
    points_3d = numpy.hstack((pts_3d, numpy.ones((N, 1))), dtype=numpy.float64)
    points_3d_proj = numpy.dot(M, points_3d.T).T
    u = points_3d_proj[:, 0] / points_3d_proj[:, 2]
    v = points_3d_proj[:, 1] / points_3d_proj[:, 2]
    residual = numpy.sum(numpy.hypot(u-pts_2d[:, 0], v-pts_2d[:, 1]), dtype=numpy.float64)
    points_3d_proj = numpy.hstack(tup=(u[:, numpy.newaxis], v[:, numpy.newaxis]), dtype=numpy.float64)
    return points_3d_proj, residual


def triangulation(M_1: numpy.ndarray, M_2: numpy.ndarray, pts_2d: numpy.ndarray) -> numpy.ndarray:
    N = pts_2d.shape[0]

    pts_3d = []

    for i in range(N):
        Q_1 = numpy.array(object=[[1, 0, -1*pts_2d[i][0]], [0, 1, -1*pts_2d[i][1]]])
        Q_2 = numpy.array(object=[[1, 0, -1*pts_2d[i][2]], [0, 1, -1*pts_2d[i][3]]])
        A_1 = Q_1@M_1
        A_2 = Q_2@M_2
        A = numpy.vstack(tup=(A_1, A_2), dtype=numpy.float64)
        _, s, V = scipy.linalg.svd(a=A)
        X = V[-1]
        X /= X[-1]
        pts_3d.append(X)

    pts_3d = numpy.vstack(tup=pts_3d, dtype=numpy.float64)

    pts_3d = pts_3d[:, :3]

    return pts_3d


def calc_resid() -> float:
    return


def visualize_3D(
        pts_3d: numpy.ndarray,
        cam_center_1: numpy.ndarray,
        cam_center_2: numpy.ndarray
) -> matplotlib.pyplot.Figure:
    matplotlib.pyplot.rc(group="font", family="serif")
    fig = matplotlib.pyplot.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(pts_3d[:, 0], pts_3d[:, 1], pts_3d[:, 2], s=35, c="orange", label='Points')
    ax.scatter(cam_center_1[0, 0], cam_center_1[1, 0], cam_center_1[2, 0], s=35, c="cyan", marker='X',
               label=r"Camera Center 1")
    ax.scatter(cam_center_2[0, 0], cam_center_2[1, 0], cam_center_2[2, 0], s=35, c="magenta", marker='X',
               label=r"Camera Center 2")
    ax.set_title(r"\textbf{Camera Centers \& Triangulated 3D Points}", usetex=True)
    ax.legend(loc="best", fontsize=10)
    ax.view_init(elev=35, azim=-45, roll=0)
    matplotlib.pyplot.show()

    return fig


def main():
    parser = argparse.ArgumentParser()
    allowed_files = ["lib", "lab"]
    parser.add_argument("--input_file", "-f", type=str, choices=allowed_files, required=True,
                        help="Input file")

    args = parser.parse_args()
    dir = args.input_file

    if dir == "lab":
        sub = "Lab"
        logger.log_info("Processing %s images" % sub)

        filepaths = config.filepaths[dir]
        pts_2d_filepath = filepaths[2]
        pts_3d_filepath = filepaths[3]

        pts_2d = numpy.loadtxt(fname=pts_2d_filepath, dtype=numpy.float64)
        pts_3d = numpy.loadtxt(fname=pts_3d_filepath, dtype=numpy.float64)

        M_1 = cam_cali(pts_3d=pts_3d, pts_2d=pts_2d[:, :2])
        M_2 = cam_cali(pts_3d=pts_3d, pts_2d=pts_2d[:, 2:])

        _, resid_1 = evaluate_points(M=M_1, pts_3d=pts_3d, pts_2d=pts_2d[:, :2])
        _, resid_2 = evaluate_points(M=M_2, pts_3d=pts_3d, pts_2d=pts_2d[:, 2:])
    elif dir == "lib":
        sub = "Library"
        logger.log_info("Processing %s images" % sub)

        filepaths = config.filepaths[dir]
        pts_2d_filepath = filepaths[2]
        M_1_filepath = filepaths[3]
        M_2_filepath = filepaths[4]

        pts_2d = numpy.loadtxt(fname=pts_2d_filepath, dtype=numpy.float64)

        M_1 = numpy.loadtxt(fname=M_1_filepath, dtype=numpy.float64)
        M_2 = numpy.loadtxt(fname=M_2_filepath, dtype=numpy.float64)
    else:
        logger.log_error("Invalid filename")

    cam_center_1 = scipy.linalg.null_space(A=M_1)
    cam_center_1 /= cam_center_1[-1]
    cam_center_2 = scipy.linalg.null_space(A=M_2)
    cam_center_2 /= cam_center_2[-1]

    logger.log_info(f"{sub} 1 Camera Projection Matrix")
    logger.log_info_raw(M_1)
    logger.log_info(f"{sub} 2 Camera Projection Matrix")
    logger.log_info_raw(M_2)
    if dir == "lab":
        logger.log_info(f"{sub} 1 Residual: %f" % resid_1)
        logger.log_info(f"{sub} 2 Residual: %f" % resid_2)
    logger.log_info(f"{sub} 1 Camera Center")
    logger.log_info_raw(cam_center_1[:-1])
    logger.log_info(f"{sub} 2 Camera Center")
    logger.log_info_raw(cam_center_2[:-1])

    pts_3d_xf = triangulation(M_1=M_1, M_2=M_2, pts_2d=pts_2d)

    fig = visualize_3D(pts_3d=pts_3d_xf, cam_center_1=cam_center_1, cam_center_2=cam_center_2)

    res_dir = os.path.join(config.res_dir, sub)
    if not os.path.exists(path=res_dir):
        os.makedirs(name=res_dir, exist_ok=True)
    if config.imwrite:
        fig.savefig(fname=os.path.join(res_dir, "cam_centers_triangulated_3d_pts.jpg"), dpi=1000, format="jpg",
                    bbox_inches="tight")

    return


if __name__ == "__main__":
    main()