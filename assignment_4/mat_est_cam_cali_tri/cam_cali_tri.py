#!/usr/bin/env python3


import argparse
import config
import logger
import matplotlib.pyplot as plt
import numpy as np
import os
import scipy.linalg


def cam_cali(pts_3d: np.ndarray, pts_2d: np.ndarray) -> np.ndarray:
    """
    https://sites.cc.gatech.edu/classes/AY2016/cs4476_fall/results/proj3/html/ckobus3/index.html
    :param pts_3d: 3D coordinates
    :param pts_2d: 2D coordinates
    :return: projection matrix
    """
    N = pts_3d.shape[0]

    zeros = np.zeros(shape=4, dtype=np.float64)
    ones = np.ones(shape=(N, 1), dtype=np.float64)
    pts_homo = np.hstack(tup=(pts_3d, ones), dtype=np.float64)

    A = []

    for i in range(N):
        row_1 = np.hstack(
            tup=(pts_homo[i], zeros, -1*pts_2d[i, 0]*pts_homo[i]),
            dtype=np.float64,
        )
        row_2 = np.hstack(
            tup=(zeros, pts_homo[i], -1*pts_2d[i, 1]*pts_homo[i]),
            dtype=np.float64,
        )
        A.append(row_1)
        A.append(row_2)

    A = np.vstack(tup=A, dtype=np.float64)

    _, _, V = scipy.linalg.svd(a=A)
    M = V[-1].reshape(3, 4)

    return M


def evaluate_points(
        M: np.ndarray,
        pts_3d: np.ndarray,
        pts_2d: np.ndarray,
) -> tuple[np.ndarray, float]:
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
    points_3d = np.hstack((pts_3d, np.ones((N, 1))), dtype=np.float64)
    points_3d_proj = np.dot(M, points_3d.T).T
    u = points_3d_proj[:, 0] / points_3d_proj[:, 2]
    v = points_3d_proj[:, 1] / points_3d_proj[:, 2]
    residual = np.sum(np.hypot(u-pts_2d[:, 0], v-pts_2d[:, 1]), dtype=np.float64)
    points_3d_proj = np.hstack(
        tup=(u[:, np.newaxis], v[:, np.newaxis]), dtype=np.float64)
    return points_3d_proj, residual


def triangulation(
        M_1: np.ndarray,
        M_2: np.ndarray,
        pts_2d: np.ndarray,
) -> np.ndarray:
    N = pts_2d.shape[0]

    pts_3d = []

    for i in range(N):
        Q_1 = np.array(object=[[1, 0, -1*pts_2d[i][0]], [0, 1, -1*pts_2d[i][1]]])
        Q_2 = np.array(object=[[1, 0, -1*pts_2d[i][2]], [0, 1, -1*pts_2d[i][3]]])
        A_1 = Q_1@M_1
        A_2 = Q_2@M_2
        A = np.vstack(tup=(A_1, A_2), dtype=np.float64)
        _, s, V = scipy.linalg.svd(a=A)
        X = V[-1]
        X /= X[-1]
        pts_3d.append(X)

    pts_3d = np.vstack(tup=pts_3d, dtype=np.float64)

    pts_3d = pts_3d[:, :3]

    return pts_3d


def calc_resid(
        M_1: np.ndarray,
        M_2: np.ndarray,
        pts_3d: np.ndarray,
        pts_2d: np.ndarray
) -> tuple[float, float]:
    N = pts_3d.shape[0]
    ones = np.ones(shape=(N, 1), dtype=np.float64)
    pts_3d = np.hstack(tup=(pts_3d, ones), dtype=np.float64)

    pts_2d_xf = []

    for i in range(N):
        x_1 = M_1@pts_3d[i]
        x_1 /= x_1[-1]
        x_2 = M_2@pts_3d[i]
        x_2 /= x_2[-1]
        x_1_x_2 = np.hstack(tup=(x_1[:2], x_2[:2]), dtype=np.float64)
        pts_2d_xf.append(x_1_x_2)

    pts_2d_xf = np.vstack(tup=pts_2d_xf, dtype=np.float64)
    loss = np.abs((pts_2d_xf-pts_2d), dtype=np.float64)
    avg_resid = np.mean(a=loss, dtype=np.float64)
    resid = np.sum(a=loss, dtype=np.float64)

    return avg_resid, resid


def visualize_3D(
        pts_3d: np.ndarray,
        cam_center_1: np.ndarray,
        cam_center_2: np.ndarray
) -> plt.Figure:
    plt.rc(group="font", family="serif")
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(
        pts_3d[:, 0],
        pts_3d[:, 1],
        pts_3d[:, 2],
        s=35,
        c="orange",
        label='Points',
    )
    ax.scatter(
        cam_center_1[0, 0],
        cam_center_1[1, 0],
        cam_center_1[2, 0],
        s=35,
        c="cyan",
        marker='X',
        label=r"Camera Center 1",
    )
    ax.scatter(
        cam_center_2[0, 0],
        cam_center_2[1, 0],
        cam_center_2[2, 0],
        s=35,
        c="magenta",
        marker='X',
        label=r"Camera Center 2",
    )
    ax.set_title(
        r"\textbf{Camera Centers \& Triangulated 3D Points}", usetex=True)
    ax.legend(loc="best", fontsize=10)
    ax.view_init(elev=35, azim=-45, roll=0)
    plt.show()

    return fig


def main():
    parser = argparse.ArgumentParser()
    allowed_files = ["lib", "lab"]
    parser.add_argument(
        "--input_file",
        "-f",
        type=str,
        choices=allowed_files,
        required=True,
        help="Input file",
    )

    args = parser.parse_args()
    dir = args.input_file

    if dir == "lab":
        sub = "Lab"
        logger.log_info("Processing %s images" % sub)

        filepaths = config.filepaths[dir]
        pts_2d_filepath = filepaths[2]
        pts_3d_filepath = filepaths[3]

        pts_2d = np.loadtxt(fname=pts_2d_filepath, dtype=np.float64)
        pts_3d = np.loadtxt(fname=pts_3d_filepath, dtype=np.float64)

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

        pts_2d = np.loadtxt(fname=pts_2d_filepath, dtype=np.float64)

        M_1 = np.loadtxt(fname=M_1_filepath, dtype=np.float64)
        M_2 = np.loadtxt(fname=M_2_filepath, dtype=np.float64)
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
        logger.log_info(f"{sub} 1 2D -> Projected 2D Residual: %f" % resid_1)
        logger.log_info(f"{sub} 2 2D -> Projected 2D Residual: %f" % resid_2)
    logger.log_info(f"{sub} 1 Camera Center")
    logger.log_info_raw(cam_center_1[:-1])
    logger.log_info(f"{sub} 2 Camera Center")
    logger.log_info_raw(cam_center_2[:-1])

    pts_3d_xf = triangulation(M_1=M_1, M_2=M_2, pts_2d=pts_2d)
    avg_resid, resid = calc_resid(
        M_1=M_1, M_2=M_2, pts_3d=pts_3d_xf, pts_2d=pts_2d)
    logger.log_info(
        f"{sub} 2D -> Triangulated 3D Average Residual: %f" % avg_resid)
    logger.log_info(f"{sub} 2D -> Triangulated 3D Total Residual: %f" % resid)

    fig = visualize_3D(
        pts_3d=pts_3d_xf, cam_center_1=cam_center_1, cam_center_2=cam_center_2)

    res_dir = os.path.join(config.res_dir, sub)
    if not os.path.exists(path=res_dir):
        os.makedirs(name=res_dir, exist_ok=True)
    if config.imwrite:
        fig.savefig(
            fname=os.path.join(res_dir, "cam_centers_triangulated_3d_pts.png"),
            dpi=1000,
            format="png",
            bbox_inches="tight",
        )

    return


if __name__ == "__main__":
    main()
