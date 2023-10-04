#!/usr/bin/env python3

import config
import cv2
import logger
import numpy
import PIL.Image

LoG_Filter = numpy.array([[0, -2, 0],
                          [-2, 8, -2],
                          [0, -2, 0]])


def rm_border(mat: numpy.ndarray, border_search_range: int, white_thres: int, black_thres: int) -> numpy.ndarray:
    """
    Remove the white and black borders of the image

    :param mat: input image matrix
    :param border_search_range: border width range to search
    :param white_thres: white pixel threshold
    :param black_thres: black pixel threshold
    :return: cropped image matrix
    """

    idx_left_list = []
    idx_right_list = []
    idx_top_list = []
    idx_btm_list = []

    # search left and right borders
    for row in mat:
        idx_l = 0
        idx_r = mat.shape[1]
        for (idx_col, pixel_val) in enumerate(row):
            if idx_col < border_search_range and (pixel_val <= black_thres or pixel_val >= white_thres):
                idx_l = max(idx_l, idx_col)
            elif idx_col >= mat.shape[1]-border_search_range and (pixel_val <= black_thres or pixel_val >= white_thres):
                idx_r = min(idx_r, idx_col)
        idx_left_list.append(idx_l)
        idx_right_list.append(idx_r)

    # search top and bottom borders
    for idx_col in range(mat.shape[1]):
        idx_t = 0
        idx_b = mat.shape[0]
        for idx_row in range(mat.shape[0]):
            pixel_val = mat[idx_row][idx_col]
            if idx_row < border_search_range and (pixel_val <= black_thres or pixel_val >= white_thres):
                idx_t = max(idx_t, idx_row)
            elif idx_row >= mat.shape[0]-border_search_range and (pixel_val <= black_thres or pixel_val >= white_thres):
                idx_b = min(idx_b, idx_row)
        idx_top_list.append(idx_t)
        idx_btm_list.append(idx_b)

    # find the coordinate that is proposed the most times
    idx_left = max(idx_left_list, key=lambda x: (x != border_search_range-1, idx_left_list.count(x)))
    idx_right = max(idx_right_list, key=lambda x: (x != mat.shape[1]-border_search_range-1, idx_right_list.count(x)))
    idx_top = max(idx_top_list, key=lambda x: (x != border_search_range-1, idx_top_list.count(x)))
    idx_btm = max(idx_btm_list, key=lambda x: (x != mat.shape[0]-border_search_range-1, idx_btm_list.count(x)))

    # crop the image with the proposed coordinate
    mat = mat[idx_top:idx_btm, idx_left:idx_right]

    return mat


def resize_image(mat: numpy.ndarray, width: int, height: int) -> numpy.ndarray:
    """
    Resize the image to the desire width and height by cropping even pixels on both side

    :param mat: input image matrix
    :param width: desire image width
    :param height: desire image height
    :return: resized image matrix
    """

    mat_h, mat_w = mat.shape

    # crop the top and bottom with even numbers of pixels
    if mat_h > height:
        h_diff = abs(mat_h - height)
        if h_diff % 2 == 0:
            mat = mat[h_diff//2:mat_h-h_diff//2, :]
        else:
            mat = mat[h_diff//2:mat_h-(h_diff//2+1), :]

    # crop the left and right with even numbers of pixels
    if mat_w > width:
        w_diff = abs(mat_w - width)
        if w_diff % 2 == 0:
            mat = mat[:, w_diff//2:mat_w-w_diff//2]
        else:
            mat = mat[:, w_diff//2:mat_w-(w_diff//2+1)]

    return mat


def split_image(mat: numpy.ndarray, border_search_range: int) -> tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]:
    """
    Split the B, G, R channels from the image by searching the black borders
    in the range of [height//3 - border_search_range, height//3 + border_search_range]

    :param mat: input image matrix
    :param border_search_range: border width range to search
    :return: B, G, R channels of the image matrix
    """

    height, width = mat.shape
    sub_img_height = height // 3
    sub_img_y_coord = []

    # search for the y coordinates to split the B, G, R channels
    for i in range(1, 3):
        y_coord_pixel_val = {}
        for j in range(i*sub_img_height-border_search_range, i*sub_img_height+border_search_range):
            y_coord_pixel_val[j] = sum(mat[j][width//2-border_search_range:width//2+border_search_range])
        sub_img_y_coord.append(min(y_coord_pixel_val, key=y_coord_pixel_val.get))

    # crop the image into B, G, R channels
    b_ch_mat = mat[0:sub_img_y_coord[0], 0:width]
    g_ch_mat = mat[sub_img_y_coord[0]:sub_img_y_coord[1], 0:width]
    r_ch_mat = mat[sub_img_y_coord[1]:height, 0:width]

    # remove the border for the B, G, R channels (only for single-scale)
    if config.high_res:
        b_ch_mat = rm_border(mat=b_ch_mat, border_search_range=5, white_thres=255, black_thres=15)
        g_ch_mat = rm_border(mat=g_ch_mat, border_search_range=5, white_thres=255, black_thres=15)
        r_ch_mat = rm_border(mat=r_ch_mat, border_search_range=5, white_thres=255, black_thres=15)

    # resize B, G, R channels to the minimum size among them
    min_height = min(b_ch_mat.shape[0], g_ch_mat.shape[0], r_ch_mat.shape[0])
    min_width = min(b_ch_mat.shape[1], g_ch_mat.shape[1], r_ch_mat.shape[1])
    b_ch_mat = resize_image(mat=b_ch_mat, width=min_width, height=min_height)
    g_ch_mat = resize_image(mat=g_ch_mat, width=min_width, height=min_height)
    r_ch_mat = resize_image(mat=r_ch_mat, width=min_width, height=min_height)

    return b_ch_mat, g_ch_mat, r_ch_mat


def preprocess_image(mat: numpy.ndarray) -> numpy.ndarray:
    """
    Preprocess image with gaussian blur and laplacian to highlight edges

    :param mat: image matrix
    :return: image matrix with edges highlighted
    """

    mat = cv2.GaussianBlur(src=mat, ksize=config.gaussian_blur_kernel_size,
                           sigmaX=config.gaussian_blur_sigmaX, sigmaY=config.gaussian_blur_sigmaY)
    mat = cv2.Laplacian(src=mat, ddepth=cv2.CV_64F, ksize=config.laplacian_kernel_size)
    return mat


def find_disp(base_ch_mat: numpy.ndarray, cmp_ch_mat: numpy.ndarray) -> tuple[int, int]:
    """
    Find the displacement of the compare channel image with respect to the base channel image with FT

    :param base_ch_mat: base channel image matrix to compare
    :param cmp_ch_mat: compare channel image matrix
    :return: displacement and best score
    """

    base_ch_mat = preprocess_image(mat=base_ch_mat)
    # base_ch_mat = cv2.filter2D(src=base_ch_mat, ddepth=-1, kernel=LoG_Filter)
    cmp_ch_mat = preprocess_image(mat=cmp_ch_mat)
    # cmp_ch_mat = cv2.filter2D(src=cmp_ch_mat, ddepth=-1, kernel=LoG_Filter)
    base_ch_ft = numpy.fft.fft2(a=base_ch_mat)
    cmp_ch_ft = numpy.fft.fft2(a=cmp_ch_mat)
    cmp_ch_ft_conj = numpy.conjugate(cmp_ch_ft)
    inv_ft = numpy.fft.ifft2(a=base_ch_ft*cmp_ch_ft_conj)
    inv_ft = numpy.fft.fftshift(x=inv_ft)
    coord = numpy.unravel_index(indices=numpy.argmax(a=inv_ft), shape=inv_ft.shape)
    disp = (coord[0] - inv_ft.shape[0]//2, coord[1] - inv_ft.shape[1]//2)

    return disp


def try_each_align(b_ch_mat: numpy.ndarray, g_ch_mat: numpy.ndarray, r_ch_mat: numpy.ndarray) -> tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]:
    """
    Find the all displacement by trying blue, green, red channel as base channel

    :param high_res: high resolution image or not
    :param b_ch_mat: blue channel image matrix
    :param g_ch_mat: green channel image matrix
    :param r_ch_mat: red channel image matrix
    :return: displacement information (base channel, displacement for first channel, displacement for second channel)
    """

    disp_g = find_disp(base_ch_mat=b_ch_mat, cmp_ch_mat=g_ch_mat)
    disp_r = find_disp(base_ch_mat=b_ch_mat, cmp_ch_mat=r_ch_mat)
    mat_b = stack_bgr_channels(b_ch_mat=b_ch_mat, g_ch_mat=g_ch_mat, r_ch_mat=r_ch_mat,
                               base_ch='B', disp_0=disp_g, disp_1=disp_r)
    print('B', disp_g, disp_r)
    disp_b = find_disp(base_ch_mat=g_ch_mat, cmp_ch_mat=b_ch_mat)
    disp_r = find_disp(base_ch_mat=g_ch_mat, cmp_ch_mat=r_ch_mat)
    mat_g = stack_bgr_channels(b_ch_mat=b_ch_mat, g_ch_mat=g_ch_mat, r_ch_mat=r_ch_mat,
                               base_ch='G', disp_0=disp_b, disp_1=disp_r)
    print('G', disp_b, disp_r)
    disp_b = find_disp(base_ch_mat=r_ch_mat, cmp_ch_mat=b_ch_mat)
    disp_g = find_disp(base_ch_mat=r_ch_mat, cmp_ch_mat=g_ch_mat)
    mat_r = stack_bgr_channels(b_ch_mat=b_ch_mat, g_ch_mat=g_ch_mat, r_ch_mat=r_ch_mat,
                               base_ch='R', disp_0=disp_b, disp_1=disp_g)
    print('R', disp_b, disp_g)

    return mat_b, mat_g, mat_r


def channel_overlap(base_ch_mat: numpy.ndarray,
                    cmp_ch_mat: numpy.ndarray,
                    disp: tuple[int, int]) -> tuple[tuple[int, int, int, int], tuple[int, int, int, int]]:
    """
    Find the overlap area between the base channel image and the compare channel image under displacement

    :param base_ch_mat: base channel image matrix
    :param cmp_ch_mat: compare channel image matrix
    :param disp: displacement of compare channel image
    :return: base channel coordinate and compare channel coordinate
    """

    base_ch_h, base_ch_w = base_ch_mat.shape
    cmp_ch_h, cmp_ch_w = cmp_ch_mat.shape
    dy, dx = disp
    base_ch_x0 = max(0, dx)
    base_ch_y0 = max(0, dy)
    base_ch_x1 = base_ch_w if dx >= 0 else base_ch_w + dx
    base_ch_y1 = base_ch_h if dy >= 0 else base_ch_h + dy
    cmp_ch_x0 = abs(min(0, dx))
    cmp_ch_y0 = abs(min(0, dy))
    cmp_ch_x1 = cmp_ch_w if dx <= 0 else cmp_ch_w - dx
    cmp_ch_y1 = cmp_ch_h if dy <= 0 else cmp_ch_h - dy

    base_ch_coord = (base_ch_x0, base_ch_y0, base_ch_x1, base_ch_y1)
    cmp_ch_coord = (cmp_ch_x0, cmp_ch_y0, cmp_ch_x1, cmp_ch_y1)

    return base_ch_coord, cmp_ch_coord


def bgr_channel_overlap(base_ch_mat: numpy.ndarray, cmp_ch_0_mat: numpy.ndarray, cmp_ch_1_mat: numpy.ndarray, disp_0: tuple[int, int], disp_1: tuple[int, int]) -> tuple[tuple[int, int, int, int], tuple[int, int, int, int], tuple[int, int, int, int]]:
    """
    Find the overlap area between base channel image and two compare channel images

    :param base_ch_mat: base channel image
    :param cmp_ch_0_mat: compare channel 0 image matrix
    :param cmp_ch_1_mat: compare channel 1 image matrix
    :param disp_0: displacement of compare channel 0 image matrix
    :param disp_1: displacement of compare channel 1 image matrix
    :return: base channel coordinate, compare channel 0 coordinate, compare channel 1 coordinate
    """

    base_ch_coord_0, cmp_ch_0_coord = channel_overlap(base_ch_mat=base_ch_mat, cmp_ch_mat=cmp_ch_0_mat, disp=disp_0)
    base_ch_coord_1, cmp_ch_1_coord = channel_overlap(base_ch_mat=base_ch_mat, cmp_ch_mat=cmp_ch_1_mat, disp=disp_1)

    # find base channel coordinate (the overlap region between two base channels)
    base_ch_x0 = max(base_ch_coord_0[0], base_ch_coord_1[0])
    base_ch_y0 = max(base_ch_coord_0[1], base_ch_coord_1[1])
    base_ch_x1 = min(base_ch_coord_0[2], base_ch_coord_1[2])
    base_ch_y1 = min(base_ch_coord_0[3], base_ch_coord_1[3])
    base_ch_coord = (base_ch_x0, base_ch_y0, base_ch_x1, base_ch_y1)

    # find compare channel 0 image shift
    cmp_ch_0_shift = (int(base_ch_coord[0] - base_ch_coord_0[0]),
                      int(base_ch_coord[1] - base_ch_coord_0[1]),
                      int(base_ch_coord[2] - base_ch_coord_0[2]),
                      int(base_ch_coord[3] - base_ch_coord_0[3]))

    # find compare channel 1 image shift
    cmp_ch_1_shift = (int(base_ch_coord[0] - base_ch_coord_1[0]),
                      int(base_ch_coord[1] - base_ch_coord_1[1]),
                      int(base_ch_coord[2] - base_ch_coord_1[2]),
                      int(base_ch_coord[3] - base_ch_coord_1[3]))

    # calculate compare channel 0 coordinate
    cmp_ch_0_coord = (int(cmp_ch_0_coord[0] + cmp_ch_0_shift[0]),
                      int(cmp_ch_0_coord[1] + cmp_ch_0_shift[1]),
                      int(cmp_ch_0_coord[2] + cmp_ch_0_shift[2]),
                      int(cmp_ch_0_coord[3] + cmp_ch_0_shift[3]))

    # calculate compare channel 1 coordinate
    cmp_ch_1_coord = (int(cmp_ch_1_coord[0] + cmp_ch_1_shift[0]),
                      int(cmp_ch_1_coord[1] + cmp_ch_1_shift[1]),
                      int(cmp_ch_1_coord[2] + cmp_ch_1_shift[2]),
                      int(cmp_ch_1_coord[3] + cmp_ch_1_shift[3]))

    return base_ch_coord, cmp_ch_0_coord, cmp_ch_1_coord


def stack_bgr_channels(b_ch_mat: numpy.ndarray, g_ch_mat: numpy.ndarray, r_ch_mat: numpy.ndarray, base_ch: str, disp_0: tuple[int, int], disp_1: tuple[int, int]) -> numpy.ndarray:
    """
    Stack B, G, R, channel to form the final RGB image

    :param b_ch_mat: blue channel image matrix
    :param g_ch_mat: green channel image matrix
    :param r_ch_mat: red channel image matrix
    :param base_ch: which color channel is used as base channel
    :param disp_0: displacement for compare channel 0 image matrix
    :param disp_1: displacement for compare channel 1 image matrix
    :return: final RGB image matrix
    """

    if base_ch not in ['B', 'G', 'R']:
        logger.log_error("Invalid base channel.")
        exit(1)

    if base_ch == 'B':
        b_ch_coord, g_ch_coord, r_ch_coord = bgr_channel_overlap(base_ch_mat=b_ch_mat,
                                                                 cmp_ch_0_mat=g_ch_mat, cmp_ch_1_mat=r_ch_mat,
                                                                 disp_0=disp_0, disp_1=disp_1)
    elif base_ch == 'G':
        g_ch_coord, b_ch_coord, r_ch_coord = bgr_channel_overlap(base_ch_mat=g_ch_mat,
                                                                 cmp_ch_0_mat=b_ch_mat, cmp_ch_1_mat=r_ch_mat,
                                                                 disp_0=disp_0, disp_1=disp_1)
    elif base_ch == 'R':
        r_ch_coord, b_ch_coord, g_ch_coord = bgr_channel_overlap(base_ch_mat=r_ch_mat,
                                                                 cmp_ch_0_mat=b_ch_mat, cmp_ch_1_mat=g_ch_mat,
                                                                 disp_0=disp_0, disp_1=disp_1)

    b_ch_mat = b_ch_mat[b_ch_coord[1]:b_ch_coord[3], b_ch_coord[0]:b_ch_coord[2]]
    g_ch_mat = g_ch_mat[g_ch_coord[1]:g_ch_coord[3], g_ch_coord[0]:g_ch_coord[2]]
    r_ch_mat = r_ch_mat[r_ch_coord[1]:r_ch_coord[3], r_ch_coord[0]:r_ch_coord[2]]

    mat = numpy.dstack(tup=(b_ch_mat, g_ch_mat, r_ch_mat))

    return mat


def align(filepath: str, high_res: bool) -> tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]:
    """
    Perform fourier-based alignment with high-res/low-res image

    :param filepath: filepath of the image
    :param high_res: process high resolution images or not
    :return: aligned image matrices, with blue, green, red each being a base channel
    """

    if high_res:
        mat = cv2.imread(filename=filepath, flags=cv2.IMREAD_GRAYSCALE)
        mat = rm_border(mat=mat, border_search_range=config.high_res_border_search_range,
                        white_thres=config.white_threshold, black_thres=config.black_threshold)
        b_ch_mat, g_ch_mat, r_ch_mat = split_image(mat=mat, border_search_range=config.high_res_border_search_range)
        mat_b, mat_g, mat_r = try_each_align(b_ch_mat=b_ch_mat, g_ch_mat=g_ch_mat, r_ch_mat=r_ch_mat)
    else:
        image = PIL.Image.open(fp=filepath)
        mat = numpy.asarray(image)
        mat = rm_border(mat=mat, border_search_range=config.low_res_border_search_range,
                        white_thres=config.white_threshold, black_thres=config.black_threshold)
        b_ch_mat, g_ch_mat, r_ch_mat = split_image(mat=mat, border_search_range=config.low_res_border_search_range)
        mat_b, mat_g, mat_r = try_each_align(b_ch_mat=b_ch_mat, g_ch_mat=g_ch_mat, r_ch_mat=r_ch_mat)

    return mat_b, mat_g, mat_r