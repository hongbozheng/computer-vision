#!/usr/bin/env python3

import align
import config
import logger
import os
import PIL.Image
import skimage


def main():
    # Single-Scale Alignment
    if config.single_scale_align:
        logger.log_info("Start single-scale alignment")
        logger.log_info("Metric = %s" % config.metric)
        single_scale_alignment_image_names = os.listdir(path=config.single_scale_alignment_images_dir)

        for filename in single_scale_alignment_image_names:
            logger.log_info("Processing file %s" % filename)
            filepath = os.path.join(config.single_scale_alignment_images_dir, filename)
            img_arr = align.single_scale_align(filepath=filepath)
            result_img_path = os.path.join(config.single_scale_alignment_results_dir, config.metric, filename)
            image = PIL.Image.fromarray(obj=img_arr)
            image.save(fp=result_img_path)

            if config.imshow:
                image.show(title="Single-Scale Alignment")

    # Multiscale Alignment
    else:
        logger.log_info("Start multiscale alignment")
        logger.log_info("Metric = %s" % config.metric)
        multiscale_alignment_image_names = os.listdir(path=config.multiscale_alignment_images_dir)

        # for filename in multiscale_alignment_image_names:
        filename = "01861a.tif"
        logger.log_info("Processing file %s" % filename)
        filepath = os.path.join(config.multiscale_alignment_images_dir, filename)
        img_arr = align.multiscale_align(filepath=filepath)
        result_img_path = os.path.join(config.multiscale_alignment_results_dir, config.metric, filename)
        skimage.io.imsave(fname=result_img_path, arr=img_arr)
        skimage.io.imshow(arr=img_arr)
        skimage.io.show()
            # break

    return


if __name__ == "__main__":
    main()