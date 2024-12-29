import config
import cv2
import logger
import matplotlib.figure
import matplotlib.pyplot as plt
import numpy as np
import scipy
import skimage.exposure
import skimage.transform


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
        shape=(max(h_0, h_1), w_0 + w_1 + thres, 3),
        fill_value=255,
        dtype=np.uint8,
    )
    canvas[:, :w_0] = mat_0
    canvas[:h_1, w_0 + thres:] = mat_1

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
    ax.plot([inliers[:, 0], inliers[:, 2] + w_0 + thres],
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


def warp_imgs(mat_0: np.ndarray, mat_1: np.ndarray, H: np.ndarray) -> np.ndarray:
    # mat_0 = mat_0.astype(dtype=numpy.float64) / 255.0
    # mat_1 = mat_1.astype(dtype=numpy.float64) / 255.0
    #
    # xf = skimage.transform.ProjectiveTransform(matrix=H)
    # h, w = mat_0.shape[:2]
    # corners = numpy.array(object=[[0, 0], [0, h], [w, 0], [h, w]], dtype=numpy.int64)
    # corners_xf = xf(coords=corners)
    #
    # corners = numpy.vstack(tup=(corners_xf, corners))
    # corner_min = numpy.min(a=corners, axis=0)
    # corner_max = numpy.max(a=corners, axis=0)
    #
    # output_shape = numpy.ceil((corner_max - corner_min)[::-1])
    #
    # offset = skimage.transform.SimilarityTransform(translation=-corner_min)
    # mat_0_warped = skimage.transform.warp(image=mat_0, inverse_map=(xf+offset).inverse, output_shape=output_shape, cval=-1)
    # mat_0_warped_0 = skimage.transform.warp(image=mat_0, inverse_map=(xf+offset).inverse, output_shape=output_shape, cval=0)
    # mat_1_warped_0 = skimage.transform.warp(image=mat_1, inverse_map=offset.inverse, output_shape=output_shape, cval=0)
    #
    # mat = mat_1_warped_0 * (mat_0_warped < 3.5e-2).astype(numpy.int8) + mat_0_warped_0 * (mat_0_warped >= 3.5e-2).astype(numpy.int8)
    # mat = (mat*255.0).astype(dtype=numpy.uint8)

    mat = cv2.warpPerspective(src=mat_0, M=H, dsize=(mat_0.shape[1], mat_0.shape[0]))

    # Reference: https://stackoverflow.com/questions/68565531/remove-black-dashed-lines-from-image-stitching
    border_mask = cv2.threshold(mat, thresh=0, maxval=255, type=cv2.THRESH_BINARY)[1]
    kernel = cv2.getStructuringElement(shape=cv2.MORPH_ELLIPSE, ksize=(5, 5))
    border_mask = cv2.morphologyEx(src=border_mask, op=cv2.MORPH_ERODE, kernel=kernel)
    mat[border_mask == 0] = 0
    border_mask = cv2.blur(src=border_mask, ksize=(5, 5))
    border_mask = skimage.exposure.rescale_intensity(
        image=border_mask,
        in_range=(127.5, 255),
        out_range=(0, 255)
    ).astype(dtype=np.float64)
    mat = (mat * border_mask + mat_1 * (255 - border_mask)) / 255
    mat = mat.clip(min=0, max=255).astype(dtype=np.uint8)

    # mat_0_mask = (mat != 0).any(axis=-1)
    # mat_1_mask = (mat_1 != 0).any(axis=-1)
    # overlap_mask = mat_0_mask & mat_1_mask
    # mat_1_mask[overlap_mask] = False
    #
    # mat[mat_1_mask] = mat_1[mat_1_mask]

    y_nz, x_nz = np.nonzero(a=mat.any(axis=2))
    mat = mat[np.min(a=y_nz):np.max(a=y_nz), np.min(a=x_nz):np.max(a=x_nz)]

    return mat


def create_canvas(mat: np.ndarray, height: int, width: int) -> np.ndarray:
    h, w, _ = mat.shape
    pad_h = height - h
    pad_w = width- w
    top = pad_h // 2
    btm = pad_h - top
    left = pad_w // 2
    right = pad_w - left
    mat = cv2.copyMakeBorder(
        src=mat,
        top=top,
        bottom=btm,
        left=left,
        right=right,
        borderType=cv2.BORDER_CONSTANT,
        value=(0, 0, 0),
    )

    return mat


def stitch_2_imgs(
        mat_0: np.ndarray,
        mat_1: np.ndarray,
        canvas_h: int,
        canvas_w: int,
) -> tuple[np.ndarray, matplotlib.figure.Figure]:
    mat_0 = create_canvas(mat=mat_0, height=canvas_h, width=canvas_w)
    mat_1 = create_canvas(mat=mat_1, height=canvas_h, width=canvas_w)

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

    H, inliners, avg_res = ransac(
        coords_0=coords_0,
        coords_1=coords_1,
        num_iters=config.ransac_num_iters,
        thres=config.ransac_thres,
    )

    mat = warp_imgs(mat_0=mat_0, mat_1=mat_1, H=H)

    mat_0 = cv2.cvtColor(src=mat_0, code=cv2.COLOR_BGR2RGB)
    mat_1 = cv2.cvtColor(src=mat_1, code=cv2.COLOR_BGR2RGB)
    fig = plt_inlier_matches(
        mat_0=mat_0, mat_1=mat_1, inliers=inliners, avg_res=avg_res)

    return mat, fig


def stitch_imgs(filepaths: list) -> tuple[np.ndarray, list[matplotlib.figure.Figure]]:
    figs = []

    if len(filepaths) > 2:
        mat = cv2.imread(filename=filepaths[0])
        for filepath in filepaths[1:]:
            mat_0 = cv2.imread(filename=filepath)
            mat, fig = stitch_2_imgs(
                mat_0=mat, mat_1=mat_0, canvas_h=1500, canvas_w=3000)
            figs.append(fig)
    else:
        filepath_0 = filepaths[0]
        filepath_1 = filepaths[1]
        mat_0 = cv2.imread(filename=filepath_0)
        mat_1 = cv2.imread(filename=filepath_1)

        mat, fig = stitch_2_imgs(
            mat_0=mat_0, mat_1=mat_1, canvas_h=1500, canvas_w=3000)
        figs.append(fig)

    return mat, figs
