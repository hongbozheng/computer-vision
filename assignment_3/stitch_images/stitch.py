import config
import cv2
import matplotlib.pyplot
import numpy
import scipy
import skimage.transform


def comp_H(coords_0: numpy.ndarray, coords_1: numpy.ndarray) -> numpy.ndarray:
    A = []

    one_vec = numpy.ones(shape=(coords_0.shape[0], 1), dtype=numpy.float64)
    coords_0 = numpy.concatenate((coords_0, one_vec), axis=1, dtype=numpy.float64)
    coords_1 = numpy.concatenate((coords_1, one_vec), axis=1, dtype=numpy.float64)

    for i in range(coords_0.shape[0]):
        row_0 = [0, 0, 0, coords_0[i][0], coords_0[i][1], coords_0[i][2], -coords_1[i][1]*coords_0[i][0],
                 -coords_1[i][1]*coords_0[i][1], -coords_1[i][1]*coords_0[i][2]]
        A.append(row_0)
        row_1 = [coords_0[i][0], coords_0[i][1], coords_0[i][2], 0, 0, 0, -coords_1[i][0]*coords_0[i][0],
                 -coords_1[i][0]*coords_0[i][1], -coords_1[i][0]*coords_0[i][2]]
        A.append(row_1)

    A = numpy.row_stack(tup=A, dtype=numpy.float64)

    _, _, V = scipy.linalg.svd(a=A, full_matrices=True)
    H = V[V.shape[0]-1]
    H = numpy.reshape(a=H, newshape=(3, 3))
    H /= H[2, 2]

    return H


def calc_residual(coords_0: numpy.ndarray, coords_1: numpy.ndarray, H: numpy.ndarray) -> numpy.ndarray:
    one_vec = numpy.ones(shape=(coords_0.shape[0], 1), dtype=numpy.float64)
    coords_0 = numpy.concatenate((coords_0, one_vec), axis=1, dtype=numpy.float64)
    coords_0 = coords_0.T
    coords_1 = coords_1.T

    coords_0_xf = H@coords_0
    coords_0_xf /= coords_0_xf[-1]
    coords_0_xf = coords_0_xf[:2]
    residuals = scipy.linalg.norm(a=(coords_1-coords_0_xf), axis=0)**2

    return residuals


def plt_inlier_matches(ax, mat_0: numpy.ndarray, mat_1: numpy.ndarray, inliers: numpy.ndarray) -> None:
    """
    Plot the matches between two images according to the matched keypoints
    :param ax: plot handle
    :param mat_0: left image
    :param mat_1: right image
    :param inliers: x,y in the first image and x,y in the second image (Nx4)
    """

    res = numpy.hstack(tup=[mat_0, mat_1])
    ax.set_aspect('equal')
    ax.imshow(X=res, cmap='gray')

    ax.plot(inliers[:, 0], inliers[:, 1], 'x', color="orange")
    ax.plot(inliers[:, 2]+mat_0.shape[1], inliers[:, 3], 'x', color="orange")
    ax.plot([inliers[:, 0], inliers[:, 2]+mat_0.shape[1]], [inliers[:, 1], inliers[:, 3]], color="orange", linewidth=0.4)
    ax.axis('off')

    matplotlib.pyplot.show()

    return


def ransac(
    coords_0: numpy.ndarray,
    coords_1: numpy.ndarray,
    num_iters: int,
    thres: float,
    mat_0: numpy.ndarray,
    mat_1: numpy.ndarray
) -> numpy.ndarray:
    max_inliners = 0
    H_best = numpy.array(object=[], dtype=numpy.float64)
    inliners_best = numpy.array(object=[], dtype=numpy.float64)

    for i in range(num_iters):
        sampled_indices = numpy.random.choice(a=coords_0.shape[0], size=4, replace=False)
        sampled_coords_0 = coords_0[sampled_indices]
        sampled_coords_1 = coords_1[sampled_indices]

        H = comp_H(coords_0=sampled_coords_0, coords_1=sampled_coords_1)

        if scipy.linalg.det(a=H) == 0:
            continue

        residuals = calc_residual(coords_0=coords_0, coords_1=coords_1, H=H)
        indices = numpy.where(residuals < thres)[0]

        if indices.shape[0] >= max_inliners:
            max_inliners = indices.shape[0]
            H_best = numpy.copy(a=H)
            coords_0_inliners = coords_0[indices]
            coords_1_inliners = coords_1[indices]
            inliners = numpy.concatenate((coords_0_inliners, coords_1_inliners), axis=1)
            inliners_best = numpy.copy(a=inliners)

    if config.plt_inliner_matches:
        fig, ax = matplotlib.pyplot.subplots(figsize=(10, 8))
        mat_0 = cv2.cvtColor(src=mat_0, code=cv2.COLOR_BGR2RGB)
        mat_1 = cv2.cvtColor(src=mat_1, code=cv2.COLOR_BGR2RGB)
        plt_inlier_matches(ax=ax, mat_0=mat_0, mat_1=mat_1, inliers=inliners_best)

    return H_best


def warp_imgs(mat_0: numpy.ndarray, mat_1: numpy.ndarray, H: numpy.ndarray) -> numpy.ndarray:
    mat_0 = mat_0.astype(dtype=numpy.float64) / 255.0
    mat_1 = mat_1.astype(dtype=numpy.float64) / 255.0

    xf = skimage.transform.ProjectiveTransform(matrix=H)
    h, w = mat_1.shape[:2]
    corners = numpy.array(object=[[0, 0], [0, h], [w, 0], [h, w]], dtype=numpy.int64)
    corners_warped = xf(coords=corners)

    corners = numpy.vstack(tup=(corners_warped, corners))
    corner_min = numpy.min(a=corners, axis=0)
    corner_max = numpy.max(a=corners, axis=0)

    output_shape = numpy.ceil((corner_max - corner_min)[::-1])

    offset = skimage.transform.SimilarityTransform(translation=-corner_min)
    mat_1_warped = skimage.transform.warp(image=mat_1, inverse_map=(xf+offset).inverse, output_shape=output_shape, cval=-1)
    mat_0_warped_0 = skimage.transform.warp(image=mat_0, inverse_map=offset.inverse, output_shape=output_shape, cval=0)
    mat_1_warped_0 = skimage.transform.warp(image=mat_1, inverse_map=(xf+offset).inverse, output_shape=output_shape, cval=0)

    mat_merged = mat_0_warped_0 * (mat_1_warped < 3.5e-2).astype(numpy.int8) + mat_1_warped_0 * (mat_1_warped >= 3.5e-2).astype(numpy.int8)
    mat = (mat_merged*255.0).astype(dtype=numpy.uint8)

    if config.imshow:
        mat_rgb = cv2.cvtColor(src=mat, code=cv2.COLOR_BGR2RGB)
        matplotlib.pyplot.figure(figsize=(15, 10))
        matplotlib.pyplot.imshow(X=mat_rgb)
        matplotlib.pyplot.show()

    return mat


def stitch_2_imgs(mat_0: numpy.ndarray, mat_1: numpy.ndarray) -> numpy.ndarray:
    mat_0_gray = cv2.cvtColor(src=mat_0, code=cv2.COLOR_BGR2GRAY)
    mat_1_gray = cv2.cvtColor(src=mat_1, code=cv2.COLOR_BGR2GRAY)

    sift_0 = cv2.SIFT_create()
    kp_0, desc_0 = sift_0.detectAndCompute(image=mat_0_gray, mask=None)
    sift_1 = cv2.SIFT_create()
    kp_1, desc_1 = sift_1.detectAndCompute(image=mat_1_gray, mask=None)

    desc_dists = scipy.spatial.distance.cdist(XA=desc_0, XB=desc_1, metric="sqeuclidean")
    row_indices, col_indices = numpy.where(desc_dists < config.dist_thres)

    coords_0 = []
    coords_1 = []
    for (r_idx, c_idx) in zip(row_indices, col_indices):
        coords_0.append(kp_0[r_idx].pt)
        coords_1.append(kp_1[c_idx].pt)

    coords_0 = numpy.row_stack(tup=coords_0, dtype=numpy.float64)
    coords_1 = numpy.row_stack(tup=coords_1, dtype=numpy.float32)

    H = ransac(coords_0=coords_0, coords_1=coords_1, num_iters=config.ransac_num_iters,
               thres=config.ransac_thres, mat_0=mat_0, mat_1=mat_1)

    mat = warp_imgs(mat_0=mat_1, mat_1=mat_0, H=H)

    return mat

def stitch_imgs(filepaths: list) -> numpy.ndarray:
    filepath_0 = filepaths[0]
    filepath_1 = filepaths[1]
    mat_0 = cv2.imread(filename=filepath_0)
    mat_1 = cv2.imread(filename=filepath_1)

    mat = stitch_2_imgs(mat_0=mat_0, mat_1=mat_1)

    return mat