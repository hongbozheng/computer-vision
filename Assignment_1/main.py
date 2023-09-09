#!/usr/bin/env python3

import basic_align
import PIL.Image

BASIC_ALIGNMENT_IMAGES = ["data/00125v.jpg", "data/00149v.jpg", "data/00153v.jpg",
                          "data/00351v.jpg", "data/00398v.jpg", "data/01112v.jpg"]
MULTISCALE_ALIGNMENT_IMAGES = ["data_hires/01047u.tif", "data_hires/01657u.tif", "data_hires/01861a.tif"]


def main():
    # Basic Alignment
    for image in BASIC_ALIGNMENT_IMAGES:
        image = PIL.Image.open(fp=image)
        image = basic_align.rm_border(image=image, width=35, white_thres=245, black_thres=35)
        image_0, image_1, image_2 = basic_align.split_image(image=image)
        disp_info = basic_align.find_best_disp(metric="NCC_EDGES", b_ch_img=image_0, g_ch_img=image_1, r_ch_img=image_2, disp_range=10)
        base_ch, disp_0, disp_1 = disp_info
        image = basic_align.stack_bgr_channels(b_ch_img=image_0, g_ch_img=image_1, r_ch_img=image_2, base_ch=base_ch, disp_0=disp_0, disp_1=disp_1)
        image.show()

    return


if __name__ == "__main__":
    main()