import config
import cv2
import numpy
import scipy
import skimage
import matplotlib.patches
import matplotlib.pyplot


def plt_blobs(mat, cx, cy, rad, color="cyan") -> None:
    """
    Draw all blobs on the image

    mat: numpy array, representing the grayscsale image
    cx, cy: numpy arrays or lists, centers of the detected blobs
    rad: numpy array or list, radius of the detected blobs

    :param mat: image matrix
    :param cx: x-coord of centers of the detected blobs
    :param cy: x-coord of centers of the detected blobs
    :param rad: radius of the detected blobs
    :param color: color of the blob
    :return: None
    """

    fig, ax = matplotlib.pyplot.subplots()
    ax.set_aspect("equal")
    ax.imshow(mat, cmap="gray")

    for x, y, r in zip(cx, cy, rad):
        circ = matplotlib.patches.Circle(xy=(x, y), radius=r, color=color, fill=False)
        ax.add_patch(circ)

    matplotlib.pyplot.title('%i circles' % len(cx))
    matplotlib.pyplot.show()

    return


def construct_blob(filepath: str, levels: int):
    mat = cv2.imread(filename=filepath)
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
        mat_gray = skimage.transform.resize(image=mat_gray, output_shape=(h//k, w//k))
        print(mat_gray.shape)
        mat_log = scipy.ndimage.filters.gaussian_laplace(input=mat_gray, sigma=1)
        mat_log = skimage.transform.resize(image=mat_log, output_shape=(h, w))
        img_pyr[k, :, :] = mat_log

    radii = numpy.empty(shape=(corner_coords.shape[0]), dtype=numpy.float32)
    corner_coords = numpy.reshape(a=corner_coords, newshape=(corner_coords.shape[0], corner_coords.shape[2]))
    for (i, (y, x)) in enumerate(corner_coords):
        radii[i] = numpy.argmax(a=img_pyr[:, x, y])+1

    mat = cv2.cvtColor(src=mat, code=cv2.COLOR_BGR2RGB)
    plt_blobs(mat=mat, cx=corner_coords[:, 0], cy=corner_coords[:, 1], rad=radii)

    return