#!/usr/bin/env python3

import config
import cv2
import logger
import matplotlib.pyplot
import numpy
import os
import PIL.Image
import stitch


def main():
    if not config.multi_imgs:
        dirs = os.listdir(path=config.img_dir)

        for dir in dirs:
            filepaths = []
            filenames = os.listdir(path=os.path.join(config.img_dir, dir))
            for filename in filenames:
                filepath = os.path.join(config.img_dir, dir, filename)
                filepaths.append(filepath)

            mat = stitch.stitch_imgs(filepaths=filepaths)
            # matplotlib.pyplot.rc(group="font", family="serif")

            if not os.path.exists(path=config.res_dir):
                os.mkdir(path=config.res_dir)

            res_img_path = os.path.join(config.res_dir, dir+".jpg")
            cv2.imwrite(filename=res_img_path, img=mat)

    return


if __name__ == "__main__":
    main()