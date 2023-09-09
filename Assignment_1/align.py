#!/usr/bin/env python3

import cv2
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

BORDER_THRESHOLD = 25
BLACK_PIXEL_THRESHOLD = 35
WHITE_PIXEL_THRESHOLD = 255


def rm_border(image: PIL.Image.Image, width: int, white_thres: int, black_thres: int) -> PIL.Image.Image:
    """
    Remove the white and black borders of the image

    :param image: input image
    :param width: border width to search
    :param white_thres: white pixel threshold
    :param black_thres: black pixel threshold
    :return: cropped image
    """

    img_arr = numpy.asarray(image)
    idx_left_list = []
    idx_right_list = []
    idx_top_list = []
    idx_btm_list = []

    # search left and right borders
    for row in img_arr:
        idx_l = 0
        idx_r = img_arr.shape[1]
        for (idx_col, pixel_val) in enumerate(row):
            if idx_col < width and (pixel_val <= black_thres or pixel_val >= white_thres):
                idx_l = max(idx_l, idx_col)
            elif idx_col >= img_arr.shape[1]-width and (pixel_val <= black_thres or pixel_val >= white_thres):
                idx_r = min(idx_r, idx_col)
        idx_left_list.append(idx_l)
        idx_right_list.append(idx_r)

    # search top and bottom borders
    for idx_col in range(img_arr.shape[1]):
        idx_t = 0
        idx_b = img_arr.shape[0]
        for idx_row in range(img_arr.shape[0]):
            pixel_val = img_arr[idx_row][idx_col]
            if idx_row < width and (pixel_val <= black_thres or pixel_val >= white_thres):
                idx_t = max(idx_t, idx_row)
            elif idx_row >= img_arr.shape[0]-width and (pixel_val <= black_thres or pixel_val >= white_thres):
                idx_b = min(idx_b, idx_row)
        idx_top_list.append(idx_t)
        idx_btm_list.append(idx_b)

    # find the coordinate that is proposed the most times
    idx_left = max(idx_left_list, key=lambda x: (x != BORDER_THRESHOLD-1, idx_left_list.count(x)))
    idx_right = max(idx_right_list, key=lambda x: (x != img_arr.shape[1]-BORDER_THRESHOLD-1, idx_right_list.count(x)))
    idx_top = max(idx_top_list, key=lambda x: (x != BORDER_THRESHOLD-1, idx_top_list.count(x)))
    idx_btm = max(idx_btm_list, key=lambda x: (x != img_arr.shape[0]-BORDER_THRESHOLD-1, idx_btm_list.count(x)))

    # crop the image with the proposed coordinate
    img_arr = img_arr[idx_top:idx_btm, idx_left:idx_right]
    image = PIL.Image.fromarray(img_arr)

    return image


def resize_image(image: PIL.Image.Image, width: int, height: int) -> PIL.Image.Image:
    """
    Resize the image to the desire width and height by cropping even pixels on both side

    :param image: input image
    :param width: desire image width
    :param height: desire image height
    :return: resized image
    """

    image_width, image_height = image.size
    image_arr = numpy.asarray(image)

    # crop the top and bottom with even numbers of pixels
    if image_height > height:
        height_diff = abs(image_height - height)
        if height_diff % 2 == 0:
            image_arr = image_arr[height_diff//2:image_height-height_diff//2, :]
        else:
            image_arr = image_arr[height_diff//2:image_height-(height_diff//2+1), :]

    # crop the left and right with even numbers of pixels
    if image_width > width:
        width_diff = abs(image_width - width)
        if width_diff % 2 == 0:
            image_arr = image_arr[:, width_diff//2:image_width-width_diff//2]
        else:
            image_arr = image_arr[:, width_diff//2:image_width-(width_diff//2+1)]

    image = PIL.Image.fromarray(obj=image_arr)
    return image


def split_image(image: PIL.Image.Image) -> tuple[PIL.Image.Image, PIL.Image.Image, PIL.Image.Image]:
    """
    Split the B, G, R channels from the image by searching the black borders
    in the range of [height//3 - BORDER_THRESHOLD, height//3 + BORDER_THRESHOLD]

    :param image: input image
    :return: B, G, R channels of the image
    """

    width, height = image.size
    sub_image_height = height // 3
    sub_image_y_coord = []
    image_arr = numpy.asarray(image)

    # search for the y coordinates to split the B, G, R channels
    for i in range(1, 3):
        y_coord_pixel_val = {}
        for j in range(i*sub_image_height-BORDER_THRESHOLD, i*sub_image_height+BORDER_THRESHOLD):
            y_coord_pixel_val[j] = sum(image_arr[j][width//2-BORDER_THRESHOLD:width//2+BORDER_THRESHOLD])
        sub_image_y_coord.append(min(y_coord_pixel_val, key=y_coord_pixel_val.get))

    # crop the image into B, G, R channels
    image_0 = image.crop(box=(0, 0, width, sub_image_y_coord[0]))
    image_1 = image.crop(box=(0, sub_image_y_coord[0], width, sub_image_y_coord[1]))
    image_2 = image.crop(box=(0, sub_image_y_coord[1], width, height))

    # remove the border for the B, G, R channels
    image_0 = rm_border(image=image_0, width=5, white_thres=255, black_thres=15)
    image_1 = rm_border(image=image_1, width=5, white_thres=255, black_thres=15)
    image_2 = rm_border(image=image_2, width=5, white_thres=255, black_thres=15)

    # resize B, G, R channels to the minimum size among them
    min_width = min(image_0.size[0], image_1.size[0], image_2.size[0])
    min_height = min(image_0.size[1], image_1.size[1], image_2.size[1])
    image_0 = resize_image(image=image_0, width=min_width, height=min_height)
    image_1 = resize_image(image=image_1, width=min_width, height=min_height)
    image_2 = resize_image(image=image_2, width=min_width, height=min_height)

    return image_0, image_1, image_2


def ncc(a: numpy.ndarray, b: numpy.ndarray) -> float:
    """
    Calculate normalized cross-correlation between two ndarray

    :param a: input ndarray
    :param b: input ndarray
    :return: normalized cross-correlation between two ndarray
    """

    return ((a/numpy.linalg.norm(a)) * (b/numpy.linalg.norm(b))).ravel().sum()


def find_disp(metric: str, base_ch_img: PIL.Image.Image, cmp_ch_img: PIL.Image, disp_range: int) -> tuple[tuple[int, int], float]:
    """
    Find the displacement of the compare channel image with respect to the base channel image

    :param metric: metric to compute the score between two images
    :param base_ch_img: base channel image to compare
    :param cmp_ch_img: compare channel image
    :param disp_range: displacement range to search
    :return: displacement and best score
    """

    if metric not in ["SSD", "SSD_EDGES", "NCC", "NCC_EDGES"]:
        print("[ERROR]: Invalid metric for finding displacements.")
        exit(1)

    best_score = float('inf') if metric == "SSD" or metric == "SSD_EDGES" else float('-inf')

    disp = (0, 0)
    base_ch_arr = numpy.asarray(base_ch_img)

    if metric == "SSD_EDGES" or "NCC_EDGES":
        base_ch_edges = cv2.Canny(image=base_ch_arr, threshold1=100, threshold2=200)

    # search for the best displacement in x-axis and y-axis
    for dy in range(-disp_range, disp_range+1):
        for dx in range(-disp_range, disp_range+1):
            cmp_ch_arr = numpy.roll(a=cmp_ch_img, shift=[dy, dx], axis=(0, 1))
            if metric == "SSD":
                score = numpy.linalg.norm((base_ch_arr - cmp_ch_arr))
                if score <= best_score:
                    best_score = score
                    disp = (dy, dx)
            elif metric == "SSD_EDGES":
                cmp_ch_edges = cv2.Canny(image=cmp_ch_arr, threshold1=100, threshold2=200)
                score = numpy.linalg.norm((base_ch_edges - cmp_ch_edges))
                if score <= best_score:
                    best_score = score
                    disp = (dy, dx)
            elif metric == "NCC":
                score = ncc(a=base_ch_arr, b=cmp_ch_arr)
                if score >= best_score:
                    best_score = score
                    disp = (dy, dx)
            elif metric == "NCC_EDGES":
                cmp_ch_edges = cv2.Canny(image=cmp_ch_arr, threshold1=100, threshold2=200)
                score = ncc(a=base_ch_edges, b=cmp_ch_edges)
                if score >= best_score:
                    best_score = score
                    disp = (dy, dx)

    return disp, best_score


def find_best_disp(metric: str, b_ch_img: PIL.Image.Image, g_ch_img: PIL.Image.Image, r_ch_img: PIL.Image.Image, disp_range: int) -> tuple[str, tuple[int, int], tuple[int, int]]:
    """
    Find the best displacement by trying blue, green, red channel as base channel

    :param metric: metric to compute the score between two images
    :param b_ch_img: blue channel image
    :param g_ch_img: green channel image
    :param r_ch_img: red channel image
    :param disp_range: displacement range to search
    :return: displacement information (base channel, displacement for first channel, displacement for second channel)
    """

    disp_map = {}
    disp_g, loss_g = find_disp(metric=metric, base_ch_img=b_ch_img, cmp_ch_img=g_ch_img, disp_range=disp_range)
    disp_r, loss_r = find_disp(metric=metric, base_ch_img=b_ch_img, cmp_ch_img=r_ch_img, disp_range=disp_range)
    disp_map[('B', disp_g, disp_r)] = loss_g + loss_r
    disp_b, loss_b = find_disp(metric=metric, base_ch_img=g_ch_img, cmp_ch_img=b_ch_img, disp_range=disp_range)
    disp_r, loss_r = find_disp(metric=metric, base_ch_img=g_ch_img, cmp_ch_img=r_ch_img, disp_range=disp_range)
    disp_map[('G', disp_b, disp_r)] = loss_b + loss_r
    disp_b, loss_b = find_disp(metric=metric, base_ch_img=r_ch_img, cmp_ch_img=b_ch_img, disp_range=disp_range)
    disp_g, loss_g = find_disp(metric=metric, base_ch_img=r_ch_img, cmp_ch_img=g_ch_img, disp_range=disp_range)
    disp_map[('R', disp_b, disp_g)] = loss_b + loss_g

    if metric == "SSD" or "SSD_EDGES":
        disp_info = min(disp_map, key=lambda k: int(disp_map[k]))
    elif metric == "NCC" or "NCC_EDGES":
        disp_info = max(disp_map, key=lambda k: int(disp_map[k]))

    return disp_info


def channel_overlap(base_ch_img: PIL.Image.Image, cmp_ch_img: PIL.Image.Image, disp: tuple[int, int]) -> tuple[tuple[int, int, int, int], tuple[int, int, int, int]]:
    """
    Find the overlap area between the base channel image and the compare channel image under displacement

    :param base_ch_img: base channel image
    :param cmp_ch_img: compare channel image
    :param disp: displacement of compare channel image
    :return: base channel coordinate and compare channel coordinate
    """

    base_ch_img_width, base_ch_img_height = base_ch_img.size
    cmp_ch_img_width, cmp_ch_img_height = cmp_ch_img.size
    dy, dx = disp
    base_ch_x0 = max(0, dx)
    base_ch_y0 = max(0, dy)
    base_ch_x1 = base_ch_img_width if dx >= 0 else base_ch_img_width + dx
    base_ch_y1 = base_ch_img_height if dy >= 0 else base_ch_img_height + dy
    cmp_ch_x0 = abs(min(0, dx))
    cmp_ch_y0 = abs(min(0, dy))
    cmp_ch_x1 = cmp_ch_img_width if dx <= 0 else cmp_ch_img_width - dx
    cmp_ch_y1 = cmp_ch_img_height if dy <= 0 else cmp_ch_img_height - dy

    base_ch_coord = (base_ch_x0, base_ch_y0, base_ch_x1, base_ch_y1)
    cmp_ch_coord = (cmp_ch_x0, cmp_ch_y0, cmp_ch_x1, cmp_ch_y1)

    return base_ch_coord, cmp_ch_coord


def bgr_channel_overlap(base_ch_img: PIL.Image.Image, cmp_ch_0_img: PIL.Image.Image, cmp_ch_1_img: PIL.Image.Image, disp_0: tuple[int, int], disp_1: tuple[int, int]) -> tuple[tuple[int, int, int, int], tuple[int, int, int, int], tuple[int, int, int, int]]:
    """
    Find the overlap area between base channel image and two compare channel images

    :param base_ch_img: base channel image
    :param cmp_ch_0_img: compare channel 0 image
    :param cmp_ch_1_img: compare channel 1 image
    :param disp_0: displacement of compare channel 0 image
    :param disp_1: displacement of compare channel 1 image
    :return: base channel coordinate, compare channel 0 coordinate, compare channel 1 coordinate
    """

    base_ch_coord_0, cmp_ch_0_coord = channel_overlap(base_ch_img=base_ch_img, cmp_ch_img=cmp_ch_0_img, disp=disp_0)
    base_ch_coord_1, cmp_ch_1_coord = channel_overlap(base_ch_img=base_ch_img, cmp_ch_img=cmp_ch_1_img, disp=disp_1)

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


def stack_bgr_channels(b_ch_img: PIL.Image.Image, g_ch_img: PIL.Image.Image, r_ch_img: PIL.Image.Image, base_ch: str, disp_0: tuple[int, int], disp_1: tuple[int, int]) -> PIL.Image.Image:
    """
    Stack B, G, R, channel to form the final RGB image

    :param b_ch_img: blue channel image
    :param g_ch_img: green channel image
    :param r_ch_img: red channel image
    :param base_ch: which color channel is used as base channel
    :param disp_0: displacement for compare channel 0 image
    :param disp_1: displacement for compare channel 1 image
    :return: final RGB image
    """

    if base_ch not in ['B', 'G', 'R']:
        print("[ERROR]: Invalid base channel.")
        exit(1)

    if base_ch == 'B':
        b_ch_coord, g_ch_coord, r_ch_coord = bgr_channel_overlap(base_ch_img=b_ch_img, cmp_ch_0_img=g_ch_img, cmp_ch_1_img=r_ch_img, disp_0=disp_0, disp_1=disp_1)
    elif base_ch == 'G':
        g_ch_coord, b_ch_coord, r_ch_coord = bgr_channel_overlap(base_ch_img=g_ch_img, cmp_ch_0_img=b_ch_img, cmp_ch_1_img=r_ch_img, disp_0=disp_0, disp_1=disp_1)
    elif base_ch == 'R':
        r_ch_coord, b_ch_coord, g_ch_coord = bgr_channel_overlap(base_ch_img=r_ch_img, cmp_ch_0_img=b_ch_img, cmp_ch_1_img=g_ch_img, disp_0=disp_0, disp_1=disp_1)

    b_ch_arr = numpy.asarray(b_ch_img)
    g_ch_arr = numpy.asarray(g_ch_img)
    r_ch_arr = numpy.asarray(r_ch_img)
    b_ch_arr = b_ch_arr[b_ch_coord[1]:b_ch_coord[3], b_ch_coord[0]:b_ch_coord[2]]
    g_ch_arr = g_ch_arr[g_ch_coord[1]:g_ch_coord[3], g_ch_coord[0]:g_ch_coord[2]]
    r_ch_arr = r_ch_arr[r_ch_coord[1]:r_ch_coord[3], r_ch_coord[0]:r_ch_coord[2]]

    image = numpy.dstack(tup=(r_ch_arr, g_ch_arr, b_ch_arr))
    image = PIL.Image.fromarray(image)

    return image