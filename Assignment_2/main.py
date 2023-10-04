#!/usr/bin/env python3

import align
import config
import cv2
import logger
import os
import PIL.Image


def main():
    # High Resolution Image Alignment
    if config.high_res:
        logger.log_info("Start multiscale alignment")
        high_res_alignment_image_names = os.listdir(path=config.high_res_images_dir)

        for filename in high_res_alignment_image_names:
            logger.log_info("Processing file %s" % filename)
            filepath = os.path.join(config.high_res_images_dir, filename)
            mat = align.align(filepath=filepath, high_res=config.high_res)
            for (i, m) in enumerate(mat):
                file_name, file_ext = os.path.splitext(p=filename)
                new_filename = file_name + '_' + config.base_ch_order[i] + file_ext
                result_img_path = os.path.join(config.ft_align_results_dir, new_filename)
                cv2.imwrite(filename=result_img_path, img=m)
                if config.imshow:
                    cv2.imshow(winname="High-Res Image Alignment", mat=mat)
                    cv2.waitKey(0)

    # Low Resolution Image Alignment
    else:
        logger.log_info("Start single-scale alignment")
        low_res_alignment_image_names = os.listdir(path=config.low_res_images_dir)

        for filename in low_res_alignment_image_names:
            logger.log_info("Processing file %s" % filename)
            filepath = os.path.join(config.low_res_images_dir, filename)
            mat = align.align(filepath=filepath, high_res=config.high_res)
            for (i, m) in enumerate(mat):
                m = m[:, :, ::-1]
                file_name, file_ext = os.path.splitext(p=filename)
                new_filename = file_name + '_' + config.base_ch_order[i] + file_ext
                result_img_path = os.path.join(config.ft_align_results_dir, new_filename)
                image = PIL.Image.fromarray(obj=m)
                image.save(fp=result_img_path)
                if config.imshow:
                    image.show(title="Low-Red Image Alignment")

    return


if __name__ == "__main__":
    main()