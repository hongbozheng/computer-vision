import config
import glob
import logger
import matplotlib.pyplot as plt
import numpy as np
import os
import scipy
import time
from PIL import Image


# Image loading and saving
def LoadFaceImages(subject_dir, subject_name, num_images):
    """
    Load the set of face images.
    The routine returns
        ambimage: image illuminated under the ambient lighting
        imarray: a 3-D array of images, h x w x Nimages
        lightdirs: Nimages x 3 array of light source directions
    """

    def load_image(fname):
        return np.asarray(Image.open(fname))

    def fname_to_ang(fname):
        yale_name = os.path.basename(fname)
        return int(yale_name[12:16]), int(yale_name[17:20])

    def sph2cart(az, el, r):
        rcos_theta = r * np.cos(el)
        x = rcos_theta * np.cos(az)
        y = rcos_theta * np.sin(az)
        z = r * np.sin(el)
        return x, y, z

    ambimage = load_image(
        os.path.join(subject_dir, subject_name + '_P00_Ambient.pgm'))
    im_list = glob.glob(os.path.join(subject_dir, subject_name + '_P00A*.pgm'))

    im_list_filtered = []
    for fname in im_list:
        mat = load_image(fname=fname)
        num_dark_pixels = np.where(mat <= config.pixel_val_thres)[0].shape[0]
        dark_pixel_ratio = num_dark_pixels/mat.size

        if dark_pixel_ratio <= config.dark_pixel_ratio_thres:
            im_list_filtered.append(fname)

    # if num_images <= len(im_list):
    #     im_sub_list = np.random.choice(im_list, num_images, replace=False)
    # else:
    #     print('Total available images is less than specified.\nProceeding with %d images.\n' % len(im_list))
    #     im_sub_list = im_list

    im_sub_list = im_list_filtered
    im_sub_list.sort()
    imarray = np.stack([load_image(fname) for fname in im_sub_list], axis=-1)
    Ang = np.array([fname_to_ang(fname) for fname in im_sub_list])

    x, y, z = sph2cart(Ang[:, 0] / 180.0 * np.pi, Ang[:, 1] / 180.0 * np.pi, 1)
    lightdirs = np.stack([y, z, x], axis=-1)

    return ambimage, imarray, lightdirs


def save_outputs(subject_name, albedo_image, surface_normals):
    im = Image.fromarray((albedo_image * 255).astype(np.uint8))
    im.save("%s_albedo.jpg" % subject_name)
    im = Image.fromarray((surface_normals[:, :, 0] * 128 + 128).astype(np.uint8))
    im.save("%s_normals_x.jpg" % subject_name)
    im = Image.fromarray((surface_normals[:, :, 1] * 128 + 128).astype(np.uint8))
    im.save("%s_normals_y.jpg" % subject_name)
    im = Image.fromarray((surface_normals[:, :, 2] * 128 + 128).astype(np.uint8))
    im.save("%s_normals_z.jpg" % subject_name)


# Plot the height map
def set_aspect_equal_3d(ax):
    """https://stackoverflow.com/questions/13685386"""
    """Fix equal aspect bug for 3D plots."""

    xlim = ax.get_xlim3d()
    ylim = ax.get_ylim3d()
    zlim = ax.get_zlim3d()
    from numpy import mean
    xmean = mean(xlim)
    ymean = mean(ylim)
    zmean = mean(zlim)
    plot_radius = max([
        abs(lim - mean_)
        for lims, mean_ in ((xlim, xmean), (ylim, ymean), (zlim, zmean))
        for lim in lims
    ])
    ax.set_xlim3d([xmean - plot_radius, xmean + plot_radius])
    ax.set_ylim3d([ymean - plot_radius, ymean + plot_radius])
    ax.set_zlim3d([zmean - plot_radius, zmean + plot_radius])


def display_output(albedo_image, height_map):
    fig = plt.figure()
    plt.imshow(albedo_image, cmap='gray')
    plt.axis('off')

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(projection='3d')
    ax.view_init(20, 20)
    X = np.arange(albedo_image.shape[0])
    Y = np.arange(albedo_image.shape[1])
    X, Y = np.meshgrid(Y, X)
    H = np.flipud(np.fliplr(height_map))
    A = np.flipud(np.fliplr(albedo_image))
    A = np.stack([A, A, A], axis=-1)
    ax.xaxis.set_ticks([])
    ax.xaxis.set_label_text('Z')
    ax.yaxis.set_ticks([])
    ax.yaxis.set_label_text('X')
    ax.zaxis.set_ticks([])
    ax.yaxis.set_label_text('Y')
    surf = ax.plot_surface(
        H, X, Y, cmap='gray', facecolors=A, linewidth=0, antialiased=False)
    set_aspect_equal_3d(ax)
    plt.show()


# Plot the surface normals
def plot_surface_normals(surface_normals):
    """
    surface_normals: h x w x 3 matrix.
    """

    fig = plt.figure()
    ax = plt.subplot(1, 3, 1)
    ax.axis('off')
    ax.set_title('X')
    im = ax.imshow(surface_normals[:, :, 0])
    ax = plt.subplot(1, 3, 2)
    ax.axis('off')
    ax.set_title('Y')
    im = ax.imshow(surface_normals[:, :, 1])
    ax = plt.subplot(1, 3, 3)
    ax.axis('off')
    ax.set_title('Z')
    im = ax.imshow(surface_normals[:, :, 2])
    plt.show()


def preprocess(ambimage, imarray):
    """
    preprocess the data:
        1. subtract ambient_image from each image in imarray.
        2. make sure no pixel is less than zero.
        3. rescale values in imarray to be between 0 and 1.
    Inputs:
        ambimage: h x w
        imarray: h x w x Nimages
    Outputs:
        processed_imarray: h x w x Nimages
    """

    ambimage = ambimage.astype(dtype=np.float64)
    imarray = imarray.astype(dtype=np.float64)
    processed_imarray = imarray - ambimage[:, :, np.newaxis]
    processed_imarray = np.clip(a=processed_imarray, a_min=0, a_max=255)
    processed_imarray /= 255.0

    return processed_imarray


def photometric_stereo(imarray, light_dirs):
    """
    Inputs:
        imarray:  h x w x Nimages
        light_dirs: Nimages x 3
    Outputs:
        albedo_image: h x w
        surface_norms: h x w x 3
    """

    h, w, _ = imarray.shape

    imarray = np.reshape(
        a=imarray,
        newshape=(imarray.shape[0]*imarray.shape[1], imarray.shape[2]),
    )
    g, res, rank, s = scipy.linalg.lstsq(a=light_dirs, b=imarray.T, cond=None)
    g = g.T
    g = np.reshape(a=g, newshape=(h, w, 3))

    albedo_image = scipy.linalg.norm(a=g, axis=2)
    surface_normals = g/albedo_image[:, :, np.newaxis]

    return albedo_image, surface_normals


def get_surface(surface_normals, integration_method):
    """
    Inputs:
        surface_normals: h x w x 3
        integration_method: string in ['average', 'column', 'row', 'random']
    Outputs:
        height_map: h x w
    """

    if integration_method not in {"average", "column", "row", "random"}:
        print("[ERROR]: Invalid integration method")
        exit()

    start_time = time.time()

    fx = surface_normals[:, :, 0]/surface_normals[:, :, 2]
    fy = surface_normals[:, :, 1]/surface_normals[:, :, 2]

    if integration_method == "average":
        h_map_rc = np.cumsum(a=fx, axis=1)[0] + np.cumsum(a=fy, axis=0)
        h_map_cr = np.cumsum(a=fx, axis=1) + \
                   np.reshape(a=np.cumsum(a=fy, axis=0)[:, 0], newshape=(-1, 1))
        height_map = (h_map_rc + h_map_cr) / 2
    elif integration_method == "row":
        height_map = np.cumsum(a=fx, axis=1)[0] + np.cumsum(a=fy, axis=0)
    elif integration_method == "column":
        height_map = np.cumsum(a=fx, axis=1) + \
                     np.reshape(a=np.cumsum(a=fy, axis=0)[:, 0], newshape=(-1, 1))
    elif integration_method == "random":
        h, w, _ = surface_normals.shape
        height_map = np.zeros(shape=(h, w))

        for y in range(h):
            for x in range(w):
                if x == 0 and y == 0:
                    continue

                for i in range(config.num_paths):
                    zeros = np.zeros(shape=x)
                    ones = np.ones(shape=y)
                    path = np.concatenate((zeros, ones))
                    np.random.shuffle(path)

                    x_steps = 0
                    y_steps = 0
                    idx = 0
                    cumsum = 0

                    while x_steps < x or y_steps < y:
                        if path[idx] == 0:
                            cumsum += fx[y_steps, x_steps]
                            x_steps += 1
                        elif path[idx] == 1:
                            cumsum += fy[y_steps, x_steps]
                            y_steps += 1
                        idx += 1

                    height_map[y, x] += cumsum
                height_map[y, x] /= config.num_paths

    end_time = time.time()
    logger.log_info("Construct height map: %ss" % str(end_time-start_time))

    return height_map


def recover_surface(subject_dir: str) -> tuple[np.ndarray, np.ndarray]:
    _, subject_name = os.path.split(p=subject_dir)
    ambient_image, imarray, light_dirs = LoadFaceImages(
        subject_dir=subject_dir,
        subject_name=subject_name,
        num_images=64,
    )
    processed_imarray = preprocess(ambimage=ambient_image, imarray=imarray)
    albedo_image, surface_normals = photometric_stereo(
        imarray=processed_imarray, light_dirs=light_dirs)

    return albedo_image, surface_normals
