#!/usr/bin/env python3

import align
import config
import logger
import os
import PIL.Image
import skimage


def main():
    # Basic Alignment
    for image_path in config.basic_alignment_image_paths:
        image = PIL.Image.open(fp=image_path)
        image = align.rm_border(image=image, width=35, white_thres=245, black_thres=35)
        b_ch_img, g_ch_img, r_ch_img = align.split_image(image=image)
        disp_info = align.find_best_disp(metric=config.metric, b_ch_img=b_ch_img, g_ch_img=g_ch_img, r_ch_img=r_ch_img, disp_range=10)
        base_ch, disp_0, disp_1 = disp_info
        image = align.stack_bgr_channels(b_ch_img=b_ch_img, g_ch_img=g_ch_img, r_ch_img=r_ch_img, base_ch=base_ch, disp_0=disp_0, disp_1=disp_1)

        _, filename = os.path.split(p=image_path)
        result_image_path = os.path.join(config.basic_alignment_results_dir, config.metric, filename)
        image.save(fp=result_image_path)

        if config.imshow:
            image.show()

    # Multiscale Alignment
    image = skimage.io.imread(fname=config.multiscale_alignment_image_paths[0])
    skimage.io.imshow(arr=image)
    skimage.io.show()

    return


if __name__ == "__main__":
    main()