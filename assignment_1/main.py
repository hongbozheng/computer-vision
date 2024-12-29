#!/usr/bin/env python3

import align
import config
import cv2
import logger
import os
import PIL.Image


def main():
    # Multiscale Alignment
    if config.img_pyr:
        logger.log_info("Start multiscale alignment")
        logger.log_info("Metric = %s" % config.metric)
        multiscale_alignment_image_names = os.listdir(
            path=config.multiscale_alignment_images_dir,
        )

        for filename in multiscale_alignment_image_names:
            logger.log_info("Processing file %s" % filename)
            filepath = os.path.join(
                config.multiscale_alignment_images_dir,
                filename,
            )
            mat = align.align(
                filepath=filepath,
                img_pyr=config.img_pyr,
                num_pyr_levels=config.num_pyramid_levels,
            )
            result_img_path = os.path.join(
                config.multiscale_alignment_results_dir,
                config.metric,
                filename,
            )
            cv2.imwrite(filename=result_img_path, img=mat)

            if config.imshow:
                cv2.imshow(winname="Multiscale Alignment", mat=mat)
                cv2.waitKey(0)

    # Single-Scale Alignment
    else:
        logger.log_info("Start single-scale alignment")
        logger.log_info("Metric = %s" % config.metric)
        single_scale_alignment_image_names = os.listdir(
            path=config.single_scale_alignment_images_dir,
        )

        for filename in single_scale_alignment_image_names:
            logger.log_info("Processing file %s" % filename)
            filepath = os.path.join(
                config.single_scale_alignment_images_dir,
                filename,
            )
            mat = align.align(
                filepath=filepath,
                img_pyr=config.img_pyr,
                num_pyr_levels=config.num_pyramid_levels,
            )
            mat = mat[:, :, ::-1]
            result_img_path = os.path.join(
                config.single_scale_alignment_results_dir,
                config.metric,
                filename,
            )
            image = PIL.Image.fromarray(obj=mat)
            image.save(fp=result_img_path)

            if config.imshow:
                image.show(title="Single-Scale Alignment")

    return


if __name__ == "__main__":
    main()
