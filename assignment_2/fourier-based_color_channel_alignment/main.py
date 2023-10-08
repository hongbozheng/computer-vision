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
        high_res_img_names = os.listdir(path=config.high_res_images_dir)
        if config.LoG_filter:
            if not os.path.exists(path=config.LoG_ft_align_results_dir):
                os.mkdir(path=config.LoG_ft_align_results_dir)
        else:
            if not os.path.exists(path=config.ft_align_results_dir):
                os.mkdir(path=config.ft_align_results_dir)

        for filename in high_res_img_names:
            for filename in low_res_img_names:
                logger.log_info("Processing file %s" % filename)
                filepath = os.path.join(config.low_res_images_dir, filename)
                mat_bgr = align.align(filepath=filepath, high_res=config.high_res)
                for (i, mat) in enumerate(mat_bgr):
                    m = mat[0]
                    inv_ft_0 = mat[1]
                    inv_ft_1 = mat[2]
                    m = m[:, :, ::-1]

                    file_name, file_ext = os.path.splitext(p=filename)
                    if config.LoG_filter:
                        res_img_dir = file_name + '_' + config.base_ch_order[i]
                        res_img_path = os.path.join(config.LoG_ft_align_results_dir, res_img_dir)
                    else:
                        res_img_dir = file_name + '_' + config.base_ch_order[i]
                        res_img_path = os.path.join(config.ft_align_results_dir, res_img_dir)
                    if not os.path.exists(path=res_img_path):
                        os.mkdir(path=res_img_path)

                    res_img_path = os.path.join(res_img_path, filename)
                    cv2.imwrite(filename=res_img_path, img=m)
                    if config.imshow:
                        cv2.imshow(winname="High-Res Image Alignment", mat=m)
                        cv2.waitKey(0)

    # Low Resolution Image Alignment
    else:
        logger.log_info("Start single-scale alignment")
        low_res_img_names = os.listdir(path=config.low_res_images_dir)

        for filename in low_res_img_names:
            logger.log_info("Processing file %s" % filename)
            filepath = os.path.join(config.low_res_images_dir, filename)
            mat_bgr = align.align(filepath=filepath, high_res=config.high_res)
            for (i, mat) in enumerate(mat_bgr):
                m = mat[0]
                inv_ft_0 = mat[1]
                inv_ft_1 = mat[2]
                m = m[:, :, ::-1]

                file_name, file_ext = os.path.splitext(p=filename)
                if config.LoG_filter:
                    res_img_dir = file_name + '_' + config.base_ch_order[i]
                    res_img_path = os.path.join(config.LoG_ft_align_results_dir, res_img_dir)
                else:
                    res_img_dir = file_name + '_' + config.base_ch_order[i]
                    res_img_path = os.path.join(config.ft_align_results_dir, res_img_dir)
                if not os.path.exists(path=res_img_path):
                    os.mkdir(path=res_img_path)

                image = PIL.Image.fromarray(obj=m)
                res_img_path = os.path.join(res_img_path, filename)
                image.save(fp=res_img_path)
                if config.imshow:
                    image.show(title="Low-Red Image Alignment")
            break

    return


if __name__ == "__main__":
    main()