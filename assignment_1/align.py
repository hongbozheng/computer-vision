#!/usr/bin/env python3

import config
import cv2
import logger
import numpy
import PIL.Image

IMAGE_BOXES = [[(27, 25, 383, 340),     # 357 x 315 (24, 25, 385, 343) 361 x 318
                (25, 361, 381, 676),    #           (24, 351, 385, 686) 361 x 335
                (26, 697, 382, 1012)],  #           (24, 693, 385, 1013) 361 x 320
               [(25, 24, 382, 339),     # 361 x 315 (24, 25, 385, 343) 361 x 318
                (24, 361, 381, 676),    #           (24, 351, 385, 686) 361 x 335
                (24, 697, 381, 1012)],  #           (24, 693, 385, 1013) 361 x 320
               [(27, 19, 381, 337),     # 354 x 318 (24, 19, 381, 337) 357 x 318
                (25, 353, 379, 671),    #           (23, 345, 380, 676) 361 x 331
                (24, 686, 378, 1004)],  #           (24, 693, 385, 1013) 361 x 320]
               [(24, 19, 374, 330),     # 350 x 311 (24, 25, 385, 343) 361 x 318
                (23, 356, 373, 667),    #           (24, 351, 385, 686) 361 x 335
                (22, 688, 372, 999)],   #           (24, 693, 385, 1013) 361 x 320
               [(20, 25, 384, 341),     # 358 x 315 (24, 25, 385, 343) 361 x 318
                (20, 361, 382, 676),    #           (24, 351, 385, 686) 361 x 335
                (20, 697, 383, 1013)],  #           (24, 693, 385, 1013) 361 x 320
              ]


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
    if config.img_pyr:
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


def ncc(mat_0: numpy.ndarray, mat_1: numpy.ndarray) -> float:
    """
    Calculate normalized cross-correlation between two ndarray

    :param mat_0: input matrix
    :param mat_1: input matrix
    :return: normalized cross-correlation between two ndarray
    """

    return ((mat_0/numpy.linalg.norm(mat_0)) * (mat_1/numpy.linalg.norm(mat_1))).ravel().sum()


def find_disp(metric: str,
              base_ch_mat: numpy.ndarray,
              cmp_ch_mat: numpy.ndarray,
              disp_range: int) -> tuple[tuple[int, int], float]:
    """
    Find the displacement of the compare channel image with respect to the base channel image

    :param metric: metric to compute the score between two images
    :param base_ch_mat: base channel image matrix to compare
    :param cmp_ch_mat: compare channel image matrix
    :param disp_range: displacement range to search
    :return: displacement and best score
    """

    best_score = float('inf') if metric == "SSD" or metric == "SSD_EDGES" else float('-inf')
    disp = (0, 0)

    if metric == "SSD_EDGES" or metric == "NCC_EDGES":
        base_ch_edges = cv2.Canny(image=base_ch_mat, threshold1=100, threshold2=200)

    # search for the best displacement in x-axis and y-axis
    for dy in range(-disp_range, disp_range+1):
        for dx in range(-disp_range, disp_range+1):
            cmp_ch_mat_shifted = numpy.roll(a=cmp_ch_mat, shift=[dy, dx], axis=(0, 1))
            if metric == "SSD":
                score = numpy.linalg.norm((base_ch_mat - cmp_ch_mat_shifted))
                if score <= best_score:
                    best_score = score
                    disp = (dy, dx)
            elif metric == "SSD_EDGES":
                cmp_ch_edges = cv2.Canny(image=cmp_ch_mat_shifted, threshold1=100, threshold2=200)
                score = numpy.linalg.norm((base_ch_edges - cmp_ch_edges))
                if score <= best_score:
                    best_score = score
                    disp = (dy, dx)
            elif metric == "NCC":
                score = ncc(mat_0=base_ch_mat, mat_1=cmp_ch_mat_shifted)
                if score >= best_score:
                    best_score = score
                    disp = (dy, dx)
            elif metric == "NCC_EDGES":
                cmp_ch_edges = cv2.Canny(image=cmp_ch_mat_shifted, threshold1=100, threshold2=200)
                score = ncc(mat_0=base_ch_edges, mat_1=cmp_ch_edges)
                if score >= best_score:
                    best_score = score
                    disp = (dy, dx)

    return disp, best_score


def blur_and_downsize(mat: numpy.ndarray) -> numpy.ndarray:
    """
    Perform gaussian blur and downsize the image by 2

    :param mat: input image array
    :return: image gaussian blurred and downsized by 2
    """

    blur_arr = cv2.GaussianBlur(src=mat, ksize=config.gaussian_blur_kernel_size,
                                sigmaX=config.gaussian_blur_sigmaX, sigmaY=config.gaussian_blur_sigmaY)
    blur_downsize_arr = cv2.resize(src=blur_arr, dsize=None, fx=1/2, fy=1/2, interpolation=cv2.INTER_AREA)
    return blur_downsize_arr


def pyr_find_disp(
    num_pyr_levels: int,
    metric: str,
    base_ch_mat: numpy.ndarray,
    cmp_ch_mat: numpy.ndarray,
    disp_range: int
) -> tuple[tuple[int, int], float]:
    """
    Image pyramid find displacement (Recursion)

    :param num_pyr_levels: number of pyramid levels
    :param metric: metric to compute the score between two image channels
    :param base_ch_mat: base channel image matrix
    :param cmp_ch_mat: compare channel image matrix
    :param disp_range: range of displacement to search for
    :return: base channel, displacement
    """

    if num_pyr_levels == 0:
        return find_disp(metric=metric, base_ch_mat=base_ch_mat, cmp_ch_mat=cmp_ch_mat, disp_range=disp_range)
    else:
        base_ch_mat = blur_and_downsize(mat=base_ch_mat)
        cmp_ch_mat = blur_and_downsize(mat=cmp_ch_mat)
        disp_prev_level, best_score_prev_level = pyr_find_disp(num_pyr_levels=num_pyr_levels-1, metric=metric,
                                                               base_ch_mat=base_ch_mat, cmp_ch_mat=cmp_ch_mat,
                                                               disp_range=disp_range)
        cmp_ch_mat = numpy.roll(a=cmp_ch_mat, shift=[disp_prev_level[0], disp_prev_level[1]], axis=[0, 1])
        disp = (disp_prev_level[0]*2, disp_prev_level[1]*2)
        disp_curr_level, best_score_curr_level = find_disp(metric=metric, base_ch_mat=base_ch_mat,
                                                           cmp_ch_mat=cmp_ch_mat, disp_range=disp_range)
        disp = (disp[0] + disp_curr_level[0], disp[1] + disp_curr_level[1])
        best_score = best_score_prev_level + best_score_curr_level
        return disp, best_score


def find_best_disp(
    img_pyr: bool,
    num_pyr_levels: int,
    metric: str,
    b_ch_mat: numpy.ndarray,
    g_ch_mat: numpy.ndarray,
    r_ch_mat: numpy.ndarray,
    disp_range: int
) -> tuple[str, tuple[int, int], tuple[int, int]]:
    """
    Find the best displacement by trying blue, green, red channel as base channel

    :param img_pyr: whether to use image pyramid or not
    :param num_pyr_levels: number of image pyramid levels
    :param metric: metric to compute the score between two image channels
    :param b_ch_mat: blue channel image matrix
    :param g_ch_mat: green channel image matrix
    :param r_ch_mat: red channel image matrix
    :param disp_range: displacement range to search
    :return: displacement information (base channel, displacement for first channel, displacement for second channel)
    """

    if metric not in ["SSD", "SSD_EDGES", "NCC", "NCC_EDGES"]:
        logger.log_error("Invalid metric for finding displacements.")
        exit(1)

    disp_map = {}

    if img_pyr:
        disp_g, score_g = pyr_find_disp(num_pyr_levels=num_pyr_levels, metric=metric,
                                        base_ch_mat=b_ch_mat, cmp_ch_mat=g_ch_mat, disp_range=disp_range)
        disp_r, score_r = pyr_find_disp(num_pyr_levels=num_pyr_levels, metric=metric,
                                        base_ch_mat=b_ch_mat, cmp_ch_mat=r_ch_mat, disp_range=disp_range)
        disp_map[('B', disp_g, disp_r)] = score_g + score_r
        disp_b, score_b = pyr_find_disp(num_pyr_levels=num_pyr_levels, metric=metric,
                                        base_ch_mat=g_ch_mat, cmp_ch_mat=b_ch_mat, disp_range=disp_range)
        disp_r, score_r = pyr_find_disp(num_pyr_levels=num_pyr_levels, metric=metric,
                                        base_ch_mat=g_ch_mat, cmp_ch_mat=r_ch_mat, disp_range=disp_range)
        disp_map[('G', disp_b, disp_r)] = score_b + score_r
        disp_b, score_b = pyr_find_disp(num_pyr_levels=num_pyr_levels, metric=metric,
                                        base_ch_mat=r_ch_mat, cmp_ch_mat=b_ch_mat, disp_range=disp_range)
        disp_g, score_g = pyr_find_disp(num_pyr_levels=num_pyr_levels, metric=metric,
                                        base_ch_mat=r_ch_mat, cmp_ch_mat=g_ch_mat, disp_range=disp_range)
        disp_map[('R', disp_b, disp_g)] = score_b + score_g
    else:
        disp_g, loss_g = find_disp(metric=metric, base_ch_mat=b_ch_mat, cmp_ch_mat=g_ch_mat, disp_range=disp_range)
        disp_r, loss_r = find_disp(metric=metric, base_ch_mat=b_ch_mat, cmp_ch_mat=r_ch_mat, disp_range=disp_range)
        disp_map[('B', disp_g, disp_r)] = loss_g + loss_r
        disp_b, loss_b = find_disp(metric=metric, base_ch_mat=g_ch_mat, cmp_ch_mat=b_ch_mat, disp_range=disp_range)
        disp_r, loss_r = find_disp(metric=metric, base_ch_mat=g_ch_mat, cmp_ch_mat=r_ch_mat, disp_range=disp_range)
        disp_map[('G', disp_b, disp_r)] = loss_b + loss_r
        disp_b, loss_b = find_disp(metric=metric, base_ch_mat=r_ch_mat, cmp_ch_mat=b_ch_mat, disp_range=disp_range)
        disp_g, loss_g = find_disp(metric=metric, base_ch_mat=r_ch_mat, cmp_ch_mat=g_ch_mat, disp_range=disp_range)
        disp_map[('R', disp_b, disp_g)] = loss_b + loss_g

    if metric == "SSD" or metric == "SSD_EDGES":
        disp_info = min(disp_map, key=lambda k: disp_map[k])
    elif metric == "NCC" or metric == "NCC_EDGES":
        disp_info = max(disp_map, key=lambda k: disp_map[k])

    return disp_info


def channel_overlap(
    base_ch_mat: numpy.ndarray,
    cmp_ch_mat: numpy.ndarray,
    disp: tuple[int, int]
) -> tuple[tuple[int, int, int, int], tuple[int, int, int, int]]:
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


def bgr_channel_overlap(
    base_ch_mat: numpy.ndarray,
    cmp_ch_0_mat: numpy.ndarray,
    cmp_ch_1_mat: numpy.ndarray,
    disp_0: tuple[int, int],
    disp_1: tuple[int, int]
) -> tuple[tuple[int, int, int, int], tuple[int, int, int, int], tuple[int, int, int, int]]:
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


def stack_bgr_channels(
    b_ch_mat: numpy.ndarray,
    g_ch_mat: numpy.ndarray,
    r_ch_mat: numpy.ndarray,
    base_ch: str,
    disp_0: tuple[int, int],
    disp_1: tuple[int, int]
) -> numpy.ndarray:
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

    img_arr = numpy.dstack(tup=(b_ch_mat, g_ch_mat, r_ch_mat))

    return img_arr


def align(filepath: str, img_pyr: bool, num_pyr_levels: int) -> numpy.ndarray:
    """
    Perform multiscale alignment (image pyramid) or single-scale alignment

    :param filepath: filepath of the image
    :param img_pyr: whether to perform image pyramid or not
    :param num_pyr_levels: number of image pyramid levels (only used if img_pyr = True)
    :return: aligned image matrix
    """

    if img_pyr:
        mat = cv2.imread(filename=filepath, flags=cv2.IMREAD_GRAYSCALE)
        mat = rm_border(mat=mat, border_search_range=config.multiscale_alignment_border_search_range,
                        white_thres=config.white_threshold, black_thres=config.black_threshold)
        b_ch_mat, g_ch_mat, r_ch_mat = split_image(mat=mat,
                                                   border_search_range=config.multiscale_alignment_border_search_range)
        disp_info = find_best_disp(img_pyr=config.img_pyr, num_pyr_levels=num_pyr_levels, metric=config.metric,
                                   b_ch_mat=b_ch_mat, g_ch_mat=g_ch_mat, r_ch_mat=r_ch_mat, disp_range=15)
        base_ch, disp_0, disp_1 = disp_info
        mat = stack_bgr_channels(b_ch_mat=b_ch_mat, g_ch_mat=g_ch_mat, r_ch_mat=r_ch_mat,
                                 base_ch=base_ch, disp_0=disp_0, disp_1=disp_1)
    else:
        image = PIL.Image.open(fp=filepath)
        mat = numpy.asarray(image)
        mat = rm_border(mat=mat, border_search_range=config.single_scale_alignment_border_search_range,
                        white_thres=config.white_threshold, black_thres=config.black_threshold)
        b_ch_mat, g_ch_mat, r_ch_mat = split_image(mat=mat,
                                                   border_search_range=config.single_scale_alignment_border_search_range)
        disp_info = find_best_disp(num_pyr_levels=num_pyr_levels, img_pyr=config.img_pyr, metric=config.metric,
                                   b_ch_mat=b_ch_mat, g_ch_mat=g_ch_mat, r_ch_mat=r_ch_mat, disp_range=10)
        base_ch, disp_0, disp_1 = disp_info
        mat = stack_bgr_channels(b_ch_mat=b_ch_mat, g_ch_mat=g_ch_mat, r_ch_mat=r_ch_mat,
                                 base_ch=base_ch, disp_0=disp_0, disp_1=disp_1)

    return mat