#!/usr/bin/env python3

from basic_align import *
import os
import PIL.Image
import skimage

BASIC_ALIGNMENT_IMAGE_PATHS = ["data/00125v.jpg", "data/00149v.jpg", "data/00153v.jpg",
                               "data/00351v.jpg", "data/00398v.jpg", "data/01112v.jpg"]
MULTISCALE_ALIGNMENT_IMAGE_PATHS = ["data_hires/01047u.tif", "data_hires/01657u.tif", "data_hires/01861a.tif"]
METRIC = "SSD"
BASIC_ALIGNMENT_RESULTS_DIR = "basic_alignment_results"
MULTISCALE_ALIGNMENT_RESULTS_DIR = "multiscale_alignment_results"
IMSHOW = False


def main():
    # Basic Alignment
    for image_path in BASIC_ALIGNMENT_IMAGE_PATHS:
        image = PIL.Image.open(fp=image_path)
        image = rm_border(image=image, width=35, white_thres=245, black_thres=35)
        b_ch_img, g_ch_img, r_ch_img = split_image(image=image)
        disp_info = find_best_disp(metric=METRIC, b_ch_img=b_ch_img, g_ch_img=g_ch_img, r_ch_img=r_ch_img, disp_range=10)
        base_ch, disp_0, disp_1 = disp_info
        image = stack_bgr_channels(b_ch_img=b_ch_img, g_ch_img=g_ch_img, r_ch_img=r_ch_img, base_ch=base_ch, disp_0=disp_0, disp_1=disp_1)

        _, filename = os.path.split(p=image_path)
        result_image_path = os.path.join(BASIC_ALIGNMENT_RESULTS_DIR, METRIC, filename)
        image.save(fp=result_image_path)

        if IMSHOW:
            image.show()

    # Multiscale Alignment
    image = skimage.io.imread(fname=MULTISCALE_ALIGNMENT_IMAGE_PATHS[0])
    skimage.io.imshow(arr=image)
    skimage.io.show()

    return


if __name__ == "__main__":
    main()