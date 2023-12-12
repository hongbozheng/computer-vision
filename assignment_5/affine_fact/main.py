#!/usr/bin/env python3

import matplotlib.pyplot
import numpy
import numpy as np
import PIL.Image
import scipy.linalg


def svd(mat: numpy.ndarray) -> tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]:
    U, W, V = scipy.linalg.svd(a=mat, full_matrices=False)

    A = U[:, :3]@numpy.sqrt(numpy.diag(v=W[:3]))
    X = numpy.sqrt(numpy.diag(v=W[:3]))@V[:3]

    X_est = A @ X

    return A, X, X_est


def calc_Q(A: numpy.ndarray) -> numpy.ndarray:
    L = scipy.linalg.pinv(a=A) @ numpy.identity(A.shape[0]) @ scipy.linalg.pinv(a=A.T)
    Q = scipy.linalg.cholesky(a=L)

    return Q


def plt_3D_structure(x: numpy.ndarray) -> None:
    x = x.T
    fig = matplotlib.pyplot.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(x[:, 0], x[:, 1], x[:, 2], color="cyan")
    ax.set(xlabel='x', ylabel='y', zlabel='z', title="3D scatter plot")
    matplotlib.pyplot.show()

    return


def calc_resid(coords_obs, coords_reproj) -> numpy.ndarray:
    x_obs = coords_obs[::2, :]
    y_obs = coords_obs[1::2, :]
    x_reproj = coords_reproj[::2, :]
    y_reproj = coords_reproj[1::2, :]

    frame_resid = numpy.sqrt((x_obs - x_reproj) ** 2 + (y_obs - y_reproj) ** 2)
    frame_resid = numpy.sum(a=frame_resid, axis=1, dtype=numpy.float64, keepdims=True)

    return frame_resid


def plt_feature_pts(meas_mat: numpy.ndarray, X_reproj: numpy.ndarray, frames: list) -> None:
    for idx in frames:
        mu = numpy.mean(a=meas_mat, axis=1, keepdims=True)
        filepath = f"data/frame00000{idx:03d}.jpg"
        mat = PIL.Image.open(fp=filepath)

        fig, ax = matplotlib.pyplot.subplots()
        ax.set_aspect("equal")
        ax.imshow(X=mat)
        ax.scatter(meas_mat[2*idx-2, :], meas_mat[2*idx-1, :], c="cyan", marker='x')
        ax.scatter(X_reproj[2*idx-2, :] + mu[2*idx-2, 0],
                X_reproj[2*idx-1, :] + mu[2*idx-1, 0], c="orange", marker='+')
        matplotlib.pyplot.show()

    return


def plt_per_frame_resid(frame_resid: numpy.ndarray) -> None:
    fig, ax = matplotlib.pyplot.subplots()
    ax.set_title("Per-frame Residuals")
    ax.plot(np.arange(1, frame_resid.shape[0] + 1), frame_resid.flatten())
    ax.set(xlabel="frame", ylabel="residual")
    matplotlib.pyplot.show()

    return


def main():
    meas_mat = numpy.loadtxt(fname="data/measurement_matrix.txt")
    mu = numpy.mean(a=meas_mat, axis=1, keepdims=True)
    X_obs = meas_mat - mu

    A, X, X_reproj = svd(mat=X_obs)
    Q = calc_Q(A=A)

    X = scipy.linalg.pinv(a=Q)@X

    plt_3D_structure(x=X)

    plt_feature_pts(meas_mat=meas_mat, X_reproj=X_reproj, frames=[35, 70, 100])

    frame_resid = calc_resid(coords_obs=X_obs, coords_reproj=X_reproj)
    plt_per_frame_resid(frame_resid=frame_resid)

    return


if __name__ == "__main__":
    main()