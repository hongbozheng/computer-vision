#!/usr/bin/env python3

import cv2
import numpy
import PIL.Image

IMAGES = ["data/00125v.jpg", "data/00149v.jpg", "data/00153v.jpg",
          "data/00351v.jpg", "data/00398v.jpg", "data/01112v.jpg"]
BORDER_THRESHOLD = 25
BLACK_PIXEL_THRESHOLD = 35
WHITE_PIXEL_THRESHOLD = 255


def rm_border(image: PIL.Image.Image, width: int, white_thres: int, black_thres: int) -> PIL.Image.Image:
    img_arr = numpy.asarray(image)
    idx_left_list = []
    idx_right_list = []
    idx_top_list = []
    idx_btm_list = []

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

    # print(idx_left_list)
    # print(idx_right_list)
    # print(idx_top_list)
    # print(idx_btm_list)
    idx_left = max(idx_left_list, key=lambda x: (x != BORDER_THRESHOLD-1, idx_left_list.count(x)))
    idx_right = max(idx_right_list, key=lambda x: (x != img_arr.shape[1]-BORDER_THRESHOLD-1, idx_right_list.count(x)))
    idx_top = max(idx_top_list, key=lambda x: (x != BORDER_THRESHOLD-1, idx_top_list.count(x)))
    idx_btm = max(idx_btm_list, key=lambda x: (x != img_arr.shape[0]-BORDER_THRESHOLD-1, idx_btm_list.count(x)))
    # idx_left = sum(idx_left_list) // len(idx_left_list)
    # idx_right = sum(idx_right_list) // len(idx_right_list)
    # idx_top = sum(idx_top_list) // len(idx_top_list)
    # idx_btm = sum(idx_btm_list) // len(idx_btm_list)
    # print(img_arr[10])
    print(idx_left, idx_right, idx_top, idx_btm)
    img_arr = img_arr[idx_top:idx_btm, idx_left:idx_right]
    image = PIL.Image.fromarray(img_arr)
    # image.show()
    return image


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


def resize_image(image: PIL.Image.Image, width: int, height: int) -> PIL.Image.Image:
    image_width, image_height = image.size
    image_arr = numpy.asarray(image)

    if image_height > height:
        height_diff = abs(image_height - height)
        if height_diff % 2 == 0:
            image_arr = image_arr[height_diff//2:image_height-height_diff//2, :]
        else:
            image_arr = image_arr[height_diff//2:image_height-(height_diff//2+1), :]

    if image_width > width:
        width_diff = abs(image_width - width)
        if width_diff % 2 == 0:
            image_arr = image_arr[:, width_diff//2:image_width-width_diff//2]
        else:
            image_arr = image_arr[:, width_diff//2:image_width-(width_diff//2+1)]

    image = PIL.Image.fromarray(obj=image_arr)
    return image


def split_image(image: PIL.Image.Image) -> tuple[PIL.Image.Image, PIL.Image.Image, PIL.Image.Image]:
    width, height = image.size
    sub_image_height = height // 3
    sub_image_y_coord = []
    image_arr = numpy.asarray(image)

    for i in range(1, 3):
        y_coord_pixel_val = {}
        for j in range(i*sub_image_height-BORDER_THRESHOLD, i*sub_image_height+BORDER_THRESHOLD):
            y_coord_pixel_val[j] = sum(image_arr[j][width//2-BORDER_THRESHOLD:width//2+BORDER_THRESHOLD])
        sub_image_y_coord.append(min(y_coord_pixel_val, key=y_coord_pixel_val.get))
    print(sub_image_y_coord)
    image_0 = image.crop(box=(0, 0, width, sub_image_y_coord[0]))
    image_1 = image.crop(box=(0, sub_image_y_coord[0], width, sub_image_y_coord[1]))
    image_2 = image.crop(box=(0, sub_image_y_coord[1], width, height))
    image_0 = rm_border(image=image_0, width=5, white_thres=255, black_thres=15)
    image_1 = rm_border(image=image_1, width=5, white_thres=255, black_thres=15)
    image_2 = rm_border(image=image_2, width=5, white_thres=255, black_thres=15)
    print(image_0.size, image_1.size, image_2.size)
    min_width = min(image_0.size[0], image_1.size[0], image_2.size[0])
    min_height = min(image_0.size[1], image_1.size[1], image_2.size[1])
    image_0 = resize_image(image=image_0, width=min_width, height=min_height)
    image_1 = resize_image(image=image_1, width=min_width, height=min_height)
    image_2 = resize_image(image=image_2, width=min_width, height=min_height)
    # image_0.show()
    # image_1.show()
    # image_2.show()
    print(image_0.size, image_1.size, image_2.size)
    return image_0, image_1, image_2

    # find_disp(image_0=image_2, image_1=image_1)
    # image_0_arr = numpy.asarray(image_0)
    # image_1_arr = numpy.asarray(image_1)
    # image_2_arr = numpy.asarray(image_2)
    # image_rgb = numpy.dstack(tup=(image_2_arr, image_1_arr, image_0_arr))
    # image_ = PIL.Image.fromarray(image_rgb)
    # image_.show()


def ncc(a: numpy.ndarray, b: numpy.ndarray):
    return ((a/numpy.linalg.norm(a)) * (b/numpy.linalg.norm(b))).ravel().sum()


def find_disp(metric: str, base_ch_img: PIL.Image.Image, cmp_ch_img: PIL.Image, disp_range: int) -> tuple[tuple[int, int], float]:
    if metric not in ["SSD", "SSD_EDGES", "NCC", "NCC_EDGES"]:
        print("[ERROR]: Invalid metric for finding displacements.")
        exit(1)

    best_score = float('inf') if metric == "SSD" or metric == "SSD_EDGES" else float('-inf')

    disp = (0, 0)
    base_ch_arr = numpy.asarray(base_ch_img)

    if metric == "SSD_EDGES" or "NCC_EDGES":
        base_ch_edges = cv2.Canny(image=base_ch_arr, threshold1=100, threshold2=200)

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

    print(disp, best_score)

    return disp, best_score


def find_best_disp(metric: str, b_ch_img: PIL.Image.Image, g_ch_img: PIL.Image.Image, r_ch_img: PIL.Image.Image, disp_range: int) -> tuple[str, tuple[int, int], tuple[int, int]]:
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

    # print(target_image_x0, target_image_y0, target_image_x1, target_image_y1)
    # print(src_image_x0, src_image_y0, src_image_x1, src_image_y1)

    base_ch_coord = (base_ch_x0, base_ch_y0, base_ch_x1, base_ch_y1)
    cmp_ch_coord = (cmp_ch_x0, cmp_ch_y0, cmp_ch_x1, cmp_ch_y1)

    return base_ch_coord, cmp_ch_coord


def bgr_channel_overlap(base_ch_img, cmp_ch_0_img, cmp_ch_1_img, disp_0, disp_1):
    base_ch_coord_0, cmp_ch_img_0_coord = channel_overlap(base_ch_img=base_ch_img, cmp_ch_img=cmp_ch_0_img, disp=disp_0)
    base_ch_coord_1, cmp_ch_img_1_coord = channel_overlap(base_ch_img=base_ch_img, cmp_ch_img=cmp_ch_1_img, disp=disp_1)
    base_ch_x0 = max(base_ch_coord_0[0], base_ch_coord_1[0])
    base_ch_y0 = max(base_ch_coord_0[1], base_ch_coord_1[1])
    base_ch_x1 = min(base_ch_coord_0[2], base_ch_coord_1[2])
    base_ch_y1 = min(base_ch_coord_0[3], base_ch_coord_1[3])
    base_ch_coord = (base_ch_x0, base_ch_y0, base_ch_x1, base_ch_y1)
    cmp_ch_0_shift = tuple(x-y for (x, y) in zip(base_ch_coord, base_ch_coord_0))
    cmp_ch_1_shift = tuple(x-y for (x, y) in zip(base_ch_coord, base_ch_coord_1))
    cmp_ch_0_coord = tuple(x+y for (x, y) in zip(cmp_ch_img_0_coord, cmp_ch_0_shift))
    cmp_ch_1_coord = tuple(x+y for (x, y) in zip(cmp_ch_img_1_coord, cmp_ch_1_shift))
    print(cmp_ch_0_shift)
    print(cmp_ch_1_shift)
    print(base_ch_coord)
    # print(target_image_coord_0)
    # print(target_image_coord_1)
    print(cmp_ch_0_coord)
    print(cmp_ch_1_coord)

    return base_ch_coord, cmp_ch_0_coord, cmp_ch_1_coord

    # (target_image_arr_width, target_image_arr_height) = target_image_arr.shape
    # (src_image_0_arr_width, src_image_0_arr_height) = src_image_0_arr.shape
    # (src_image_1_arr_width, src_image_1_arr_height) = src_image_1_arr.shape
    # min_width = min(target_image_arr_width, src_image_0_arr_width, src_image_1_arr_width)
    # min_height = min(target_image_arr_height, src_image_0_arr_height, src_image_1_arr_height)
    # print(target_image_arr.shape, src_image_0_arr.shape, src_image_1_arr.shape)
    # print(min_width, min_height)
    # target_image_arr = target_image_arr[:min_width, :min_height]
    # src_image_0_arr = src_image_0_arr[:min_width, :min_height]
    # src_image_1_arr = src_image_1_arr[:min_width, :min_height]

    # exit(1)
    # src_image_0 = numpy.roll(a=src_image_0, shift=disp_0, axis=(0, 1))
    # src_image_1 = numpy.roll(a=src_image_1, shift=disp_1, axis=(0, 1))
    # target_image_arr = numpy.asarray(target_image)
    # src_image_0_arr = numpy.asarray(src_image_0)
    # src_image_1_arr = numpy.asarray(src_image_1)

    # image_0_width, image_0_height = image_0.size
    # image_1_width, image_1_height = image_1.size
    #
    # image_0_arr = numpy.asarray(image_0)
    # image_1_arr = numpy.asarray(image_1)
    #
    # score = math.inf
    # disp = (0, 0)
    #
    # for dy in range(-15, 15):
    #     for dx in range(-15, 15):
    #         print(dy, dx)
    #         image_0_x0 = max(0, dy)
    #         image_0_y0 = max(0, dx)
    #         image_0_x1 = min(image_0_width, image_0_width + dy)
    #         image_0_y1 = image_1_height + dx if image_1_height + dx <= image_0_height else image_0_height
    #         print(image_0_x0, image_0_y0, image_0_x1, image_0_y1)
    #         image_1_x0 = abs(dy) if dy <= 0 else 0
    #         image_1_y0 = abs(dx) if dx <= 0 else 0
    #         image_1_x1 = image_1_width if dy <= 0 else image_1_width - dy
    #         image_1_y1 = image_1_height if image_1_height + dx <= image_0_height else image_0_height - dx
    #         print(image_1_x0, image_1_y0, image_1_x1, image_1_y1)
    #         cur_score = numpy.linalg.norm(image_0_arr[image_0_y0:image_0_y1, image_0_x0:image_0_x1] - image_1_arr[image_1_y0:image_1_y1, image_1_x0:image_1_x1])
    #
    #         cur_score += ((image_0_width - (image_0_y1 - image_0_y0)) + (image_0_height - (image_0_x1 - image_0_x0))) * 200
    #
    #         if cur_score < score:
    #             score = cur_score
    #             disp = (dy, dx)
    #         print(cur_score)
    #
    # print("final disp", disp)


def stack_bgr_channels(b_ch_img, g_ch_img, r_ch_img, base_ch, disp_0, disp_1):
    if base_ch not in ['B', 'G', 'R']:
        print("[ERROR]: Invalid base channel.")
        exit(1)

    if base_ch == 'B':
        b_ch_coord, g_ch_coord, r_ch_coord = bgr_channel_overlap(base_ch_img=b_ch_img, cmp_ch_0_img=g_ch_img, cmp_ch_1_img=r_ch_img, disp_0=disp_0, disp_1=disp_1)
        pass
    elif base_ch == 'G':
        g_ch_coord, b_ch_coord, r_ch_coord = bgr_channel_overlap(base_ch_img=g_ch_img, cmp_ch_0_img=b_ch_img, cmp_ch_1_img=r_ch_img, disp_0=disp_0, disp_1=disp_1)
        pass
    elif base_ch == 'R':
        r_ch_coord, b_ch_coord, g_ch_coord = bgr_channel_overlap(base_ch_img=r_ch_img, cmp_ch_0_img=b_ch_img, cmp_ch_1_img=g_ch_img, disp_0=disp_0, disp_1=disp_1)
        pass

    b_ch_arr = numpy.asarray(b_ch_img)
    print(b_ch_arr.shape)
    g_ch_arr = numpy.asarray(g_ch_img)
    r_ch_arr = numpy.asarray(r_ch_img)
    b_ch_arr = b_ch_arr[b_ch_coord[1]:b_ch_coord[3], b_ch_coord[0]:b_ch_coord[2]]
    g_ch_arr = g_ch_arr[g_ch_coord[1]:g_ch_coord[3], g_ch_coord[0]:g_ch_coord[2]]
    r_ch_arr = r_ch_arr[r_ch_coord[1]:r_ch_coord[3], r_ch_coord[0]:r_ch_coord[2]]

    print(b_ch_arr.shape, g_ch_arr.shape, r_ch_arr.shape)
    image = numpy.dstack(tup=(r_ch_arr, g_ch_arr, b_ch_arr))
    image = PIL.Image.fromarray(image)
    image.show()

    return


def main():
    for image in IMAGES:
        image = PIL.Image.open(fp=image)
        image = rm_border(image=image, width=35, white_thres=245, black_thres=35)
        image_0, image_1, image_2 = split_image(image=image)

        # disp_0, _ = find_disp(metric="NCC", base_ch_img=image_0, cmp_ch_img=image_1, disp_range=15)
        # disp_1, _ = find_disp(metric="NCC", base_ch_img=image_0, cmp_ch_img=image_2, disp_range=15)

        disp_info = find_best_disp(metric="SSD", b_ch_img=image_0, g_ch_img=image_1, r_ch_img=image_2, disp_range=10)
        print(disp_info)

        base_ch, disp_0, disp_1 = disp_info
        stack_bgr_channels(b_ch_img=image_0, g_ch_img=image_1, r_ch_img=image_2, base_ch=base_ch, disp_0=disp_0, disp_1=disp_1)

    # image_0_arr = numpy.asarray(image_0)
    # image_1_arr = numpy.asarray(image_1)
    # image_2_arr = numpy.asarray(image_2)
    # image_1_arr = numpy.roll(a=image_1_arr, shift=disp_0, axis=(0, 1))
    # image_2_arr = numpy.roll(a=image_2_arr, shift=disp_1, axis=(0, 1))
    # image = numpy.dstack(tup=(image_2_arr, image_1_arr, image_0_arr))
    # image = PIL.Image.fromarray(image)
    # image.show()


    # image = PIL.Image.open(fp=IMAGES[0])
    # image_0 = image.crop(box=IMAGE_BOXES[0][0])
    # image_1 = image.crop(box=IMAGE_BOXES[0][1])
    # image_2 = image.crop(box=IMAGE_BOXES[0][2])
    # image_0.show()
    # image_1.show()
    # image_2.show()
    # image_0_arr = numpy.asarray(image_0)
    # image_1_arr = numpy.asarray(image_1)
    # image_2_arr = numpy.asarray(image_2)
    # image_rgb = numpy.dstack(tup=(image_2_arr, image_1_arr, image_0_arr))
    # image_ = PIL.Image.fromarray(image_rgb)
    # image_.show()

    # image = image_preprocessing(image=image)
    # image.show()
    # for name in IMAGES:
    #     image = PIL.Image.open(name)
    #     print(image.size, image.width, image.height)
    #     image = image_preprocessing(image)
    #     img_arr = numpy.asarray(image)
    #     sub_image_width = img_arr.shape[0] // 3
    #     image_0_arr = img_arr[:sub_image_width, :]
    #     image_1_arr = img_arr[sub_image_width:2*sub_image_width, :]
    #     image_2_arr = img_arr[2*sub_image_width:, :]
    #     image_0 = PIL.Image.fromarray(image_0_arr)
    #     image_1 = PIL.Image.fromarray(image_1_arr)
    #     image_2 = PIL.Image.fromarray(image_2_arr)
    #     image_0.show()
    #     image_1.show()
    #     image_2.show()
    #     print(image_0.size, image_1.size, image_2.size)
    #     break
    return

if __name__ == "__main__":
    main()