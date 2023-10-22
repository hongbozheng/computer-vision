#!/usr/bin/env python3

import config
import cv2
import logger
import matplotlib.pyplot
import os
import stitch


def main():
    if config.multi_imgs:
        logger.log_info("Start stitching multiple images")
        dirs = os.listdir(path=config.img_multi_dir)

        for dir in dirs:
            logger.log_info("Processing image %s" % dir)
            filepaths = []
            filenames = os.listdir(path=os.path.join(config.img_multi_dir, dir))
            for filename in filenames:
                filepath = os.path.join(config.img_multi_dir, dir, filename)
                filepaths.append(filepath)

            mat = stitch.stitch_imgs(filepaths=filepaths)

            os.makedirs(name=config.res_dir, exist_ok=True)
            res_img_path = os.path.join(config.res_dir, dir+".jpg")
            cv2.imwrite(filename=res_img_path, img=mat)

            if config.imshow:
                mat = cv2.cvtColor(src=mat, code=cv2.COLOR_BGR2RGB)
                fig, ax = matplotlib.pyplot.subplots(figsize=(15, 10))
                ax.imshow(X=mat)
                ax.set_title("Stitched Image")
                matplotlib.pyplot.show()

    else:
        logger.log_info("Start stitching 2 images")
        dirs = os.listdir(path=config.img_dir)

        for dir in dirs:
            logger.log_info("Processing image %s" % dir)
            filepaths = []
            filenames = os.listdir(path=os.path.join(config.img_dir, dir))
            for filename in filenames:
                filepath = os.path.join(config.img_dir, dir, filename)
                filepaths.append(filepath)

            mat = stitch.stitch_imgs(filepaths=filepaths)

            os.makedirs(name=config.res_dir, exist_ok=True)
            res_img_path = os.path.join(config.res_dir, dir+".jpg")
            cv2.imwrite(filename=res_img_path, img=mat)

            if config.imshow:
                mat = cv2.cvtColor(src=mat, code=cv2.COLOR_BGR2RGB)
                fig, ax = matplotlib.pyplot.subplots(figsize=(15, 10))
                ax.imshow(X=mat)
                ax.set_title("Stitched Image")
                matplotlib.pyplot.show()

    return


if __name__ == "__main__":
    main()