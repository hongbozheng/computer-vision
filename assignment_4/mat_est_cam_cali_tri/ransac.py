import config
import cv2
import logger
import matplotlib.figure
import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg


def comp_matches(
        mat_0: np.ndarray,
        mat_1: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    mat_0_gray = cv2.cvtColor(src=mat_0, code=cv2.COLOR_BGR2GRAY)
    mat_1_gray = cv2.cvtColor(src=mat_1, code=cv2.COLOR_BGR2GRAY)

    sift_0 = cv2.SIFT_create()
    kp_0, desc_0 = sift_0.detectAndCompute(image=mat_0_gray, mask=None)
    sift_1 = cv2.SIFT_create()
    kp_1, desc_1 = sift_1.detectAndCompute(image=mat_1_gray, mask=None)

    desc_dists = scipy.spatial.distance.cdist(
        XA=desc_0, XB=desc_1, metric="sqeuclidean")
    row_indices, col_indices = np.where(desc_dists < config.dist_thres)

    coords_0 = []
    coords_1 = []
    for (r_idx, c_idx) in zip(row_indices, col_indices):
        coords_0.append(kp_0[r_idx].pt)
        coords_1.append(kp_1[c_idx].pt)

    coords_0 = np.row_stack(tup=coords_0, dtype=np.float64)
    coords_1 = np.row_stack(tup=coords_1, dtype=np.float64)

    return coords_0, coords_1


def comp_H(coords_0: np.ndarray, coords_1: np.ndarray) -> np.ndarray:
    A = []

    one_vec = np.ones(shape=(coords_0.shape[0], 1), dtype=np.float64)
    coords_0 = np.concatenate((coords_0, one_vec), axis=1, dtype=np.float64)
    coords_1 = np.concatenate((coords_1, one_vec), axis=1, dtype=np.float64)

    for i in range(coords_0.shape[0]):
        row_0 = [0, 0, 0, coords_0[i][0], coords_0[i][1], coords_0[i][2], -coords_1[i][1]*coords_0[i][0],
                 -coords_1[i][1]*coords_0[i][1], -coords_1[i][1]*coords_0[i][2]]
        A.append(row_0)
        row_1 = [coords_0[i][0], coords_0[i][1], coords_0[i][2], 0, 0, 0, -coords_1[i][0]*coords_0[i][0],
                 -coords_1[i][0]*coords_0[i][1], -coords_1[i][0]*coords_0[i][2]]
        A.append(row_1)

    A = np.row_stack(tup=A, dtype=np.float64)

    _, _, V = scipy.linalg.svd(a=A, full_matrices=True)
    H = V[V.shape[0]-1]
    H = np.reshape(a=H, newshape=(3, 3))
    H /= H[2, 2]

    return H


def calc_residual(
        coords_0: np.ndarray,
        coords_1: np.ndarray,
        H: np.ndarray,
) -> np.ndarray:
    one_vec = np.ones(shape=(coords_0.shape[0], 1), dtype=np.float64)
    coords_0 = np.concatenate((coords_0, one_vec), axis=1, dtype=np.float64)
    coords_0 = coords_0.T
    coords_1 = coords_1.T

    coords_0_xf = H@coords_0

    if (coords_0_xf[-1] == 0).any():
        return np.full(shape=coords_0_xf.shape[1], fill_value=config.ransac_thres)

    coords_0_xf /= coords_0_xf[-1]
    coords_0_xf = coords_0_xf[:2]
    residuals = scipy.linalg.norm(a=(coords_1-coords_0_xf), axis=0)**2

    return residuals


def plt_inlier_matches(
        mat_0: np.ndarray,
        mat_1: np.ndarray,
        inliers: np.ndarray,
        avg_res: float,
) -> matplotlib.figure.Figure:
    h_0, w_0, _ = mat_0.shape
    h_1, w_1, _ = mat_1.shape
    thres = 35

    canvas = np.full(
        shape=(max(h_0, h_1), w_0+w_1+thres, 3), fill_value=255, dtype=np.uint8)
    canvas[:, :w_0] = mat_0
    canvas[:h_1, w_0+thres:] = mat_1

    plt.rc(group="font", family="serif")
    fig, ax = plt.subplots(figsize=(15, 10))
    ax.set_aspect("equal")
    ax.imshow(X=canvas)
    ax.plot(inliers[:, 0], inliers[:, 1], 'x', color="orange", markersize=5)
    ax.plot(
        inliers[:, 2] + w_0 + thres,
        inliers[:, 3],
        'x',
        color="orange",
        markersize=5,
    )
    ax.plot(
        [inliers[:, 0], inliers[:, 2] + w_0+thres],
        [inliers[:, 1], inliers[:, 3]],
        color="cyan",
        linewidth=0.50,
    )
    ax.axis("off")
    ax.set_title("%s Inliner Matches     Average Residual = %s" %
                 (str(inliers.shape[0]), str(avg_res)))

    if config.plt_inliner_matches:
        plt.show()

    return fig


def ransac(
    coords_0: np.ndarray,
    coords_1: np.ndarray,
    num_iters: int,
    thres: float
) -> tuple[np.ndarray, np.ndarray, float]:
    max_inliners = 0
    H_best = np.array(object=[], dtype=np.float64)
    inliners_best = np.array(object=[], dtype=np.float64)

    for i in range(num_iters):
        sampled_indices = np.random.choice(
            a=coords_0.shape[0], size=4, replace=False)
        sampled_coords_0 = coords_0[sampled_indices]
        sampled_coords_1 = coords_1[sampled_indices]

        H = comp_H(coords_0=sampled_coords_0, coords_1=sampled_coords_1)

        if scipy.linalg.det(a=H) == 0:
            continue

        residuals = calc_residual(coords_0=coords_0, coords_1=coords_1, H=H)
        indices = np.where(residuals < thres)[0]

        if indices.shape[0] >= max_inliners:
            max_inliners = indices.shape[0]
            H_best = np.copy(a=H)
            coords_0_inliners = coords_0[indices]
            coords_1_inliners = coords_1[indices]
            inliners = np.concatenate(
                (coords_0_inliners, coords_1_inliners), axis=1)
            inliners_best = np.copy(a=inliners)
            avg_res = np.sum(a=residuals[indices]) / inliners.shape[0]

    logger.log_info("Num of inliners:  %d" % inliners.shape[0])
    logger.log_info("Average residual: %f" % avg_res)

    return H_best, inliners_best, avg_res
