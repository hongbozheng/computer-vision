import config
import cv2
import logger
import numpy
import scipy
import skimage
import matplotlib.patches
import matplotlib.pyplot


def mark_corners_plt_blobs_orien(
    mat: numpy.ndarray,
    cx: numpy.ndarray,
    cy: numpy.ndarray,
    mark_size: int,
    x_grad: numpy.ndarray,
    y_grad: numpy.ndarray,
    rad: numpy.ndarray,
    arrow_head_w: int,
    arrow_head_l: int,
    mark_color: str,
    blob_color: str,
    arrow_color: str
) -> matplotlib.pyplot.subplot:
    """
    Mark all corners with 'x', plot all blobs, and the orientations of the window

    mat: numpy array, representing the grayscsale image
    cx, cy: numpy arrays or lists, centers of the detected blobs
    rad: numpy array or list, radius of the detected blobs

    :param mat: image matrix
    :param cx: x-coord of centers of the detected blobs
    :param cy: x-coord of centers of the detected blobs
    :param mark_size: size of the mark
    :param x_grad: gradient in x-direction
    :param y_grad: gradient in y-direction
    :param rad: radius of the detected blobs
    :param arrow_head_w: total width of the full arrow head
    :param arrow_head_l: length of arrow head
    :param mark_color: color of mark (corners)
    :param blob_color: color of blobs
    :param arrow_color: color of arrows
    :return: fig, ax: matplotlib.pyplot.subplot
    """

    fig, ax = matplotlib.pyplot.subplots()
    ax.set_aspect("equal")
    ax.imshow(mat, cmap="gray")

    ax.scatter(x=cx, y=cy, s=mark_size, color=mark_color, marker='x')
    for (x, y, r) in zip(cx, cy, rad):
        circle = matplotlib.patches.Circle(xy=(x, y), radius=r, color=blob_color, fill=False)
        ax.add_patch(circle)
    for (x, y, dx, dy) in zip(cx, cy, x_grad, y_grad):
        ax.arrow(x=x, y=y, dx=dx, dy=dy, head_width=arrow_head_w, head_length=arrow_head_l, color=arrow_color)

    matplotlib.pyplot.title('%i circles' % len(cx))

    return fig, ax


def construct_blob(mat: numpy.ndarray, levels: int) -> matplotlib.pyplot.subplot:
    mat_gray = cv2.cvtColor(src=mat, code=cv2.COLOR_BGR2GRAY)
    mat_gray = mat_gray / 255.0
    mat_gray = mat_gray.astype(dtype=numpy.float32)
    corner_coords = cv2.goodFeaturesToTrack(image=mat_gray, maxCorners=config.max_corners,
                                            qualityLevel=config.quality_level, minDistance=config.min_dist,
                                            blockSize=config.blk_size, useHarrisDetector=config.harris, k=config.k)
    corner_coords = corner_coords.astype(dtype=numpy.int16)

    h, w = mat_gray.shape
    img_pyr = numpy.empty(shape=[levels, h, w], dtype=numpy.float32)

    for k in range(1, levels):
        mat_ds = skimage.transform.resize(image=mat_gray, output_shape=(h//k, w//k))
        print(mat_ds.shape)
        mat_log = scipy.ndimage.filters.gaussian_laplace(input=mat_ds, sigma=1)
        mat_log = skimage.transform.resize(image=mat_log, output_shape=(h, w))
        img_pyr[k, :, :] = mat_log

    radii = numpy.empty(shape=(corner_coords.shape[0]), dtype=numpy.float32)
    corner_coords = numpy.reshape(a=corner_coords, newshape=(corner_coords.shape[0], corner_coords.shape[2]))
    for (i, (y, x)) in enumerate(corner_coords):
        radii[i] = numpy.argmax(a=img_pyr[:, x, y])+1

    x_grad = cv2.Sobel(mat_gray, cv2.CV_64F, dx=1, dy=0, ksize=config.ksize)
    y_grad = cv2.Sobel(mat_gray, cv2.CV_64F, dx=0, dy=1, ksize=config.ksize)
    x_grad = x_grad[corner_coords[:, 1], corner_coords[:, 0]]
    y_grad = y_grad[corner_coords[:, 1], corner_coords[:, 0]]

    mat = cv2.cvtColor(src=mat, code=cv2.COLOR_BGR2RGB)
    fig, ax = mark_corners_plt_blobs_orien(mat=mat, cx=corner_coords[:, 0], cy=corner_coords[:, 1],
                                           mark_size=config.mark_size, rad=radii, x_grad=x_grad, y_grad=y_grad,
                                           arrow_head_w=config.arrow_head_w, arrow_head_l=config.arrow_head_l,
                                           mark_color=config.mark_color, blob_color=config.blob_color,
                                           arrow_color=config.arrow_color)

    return fig, ax


def xform(filepath: str, levels: int, xf: str) -> matplotlib.pyplot.subplot:
    """
    Perform transform on the original image

    :param filepath: filepath of the input image
    :param levels: number of image pyramid levels
    :param xf: transform type
    :return: matplotlib.pyplot.subplot
    """

    if xf not in {"orig", "sl", "sr", "rccw", "rcw", "x2"}:
        logger.log_error("Invalid transform type.")

    mat = cv2.imread(filename=filepath)

    if xf == "orig":
        fig, ax = construct_blob(mat=mat, levels=levels)
    elif xf == "sl":
        mat_sl = numpy.roll(a=mat, shift=-mat.shape[1]//5, axis=1)
        print(mat_sl.shape)
        mat_sl = mat_sl[:, :mat.shape[1]-mat.shape[1]//5, :]
        print(mat_sl.shape)
        exit()
        canvas[:, 0:img.shape[1] - img.shape[1] // 5, :] = img[:, 0:img.shape[1] - img.shape[1] // 5, :]
        img = canvas
    elif xf == "sr":
        pass
    elif xf == "rccw":
        mat = cv2.rotate(src=mat, rotateCode=cv2.ROTATE_90_COUNTERCLOCKWISE)
        fig, ax = construct_blob(mat=mat, levels=levels)
    elif xf == "rcw":
        mat = cv2.rotate(src=mat, rotateCode=cv2.ROTATE_90_CLOCKWISE)
        fig, ax = construct_blob(mat=mat, levels=levels)
    elif xf == "x2":
        mat_x2 = cv2.resize(src=mat, dsize=(mat.shape[1]*2, mat.shape[0]*2))
        mat_x2 = mat_x2[mat.shape[0]//2:mat.shape[0]//2+mat.shape[0], mat.shape[1]//2:mat.shape[1]//2+mat.shape[1]]
        fig, ax = construct_blob(mat=mat_x2, levels=levels)

    return fig, ax