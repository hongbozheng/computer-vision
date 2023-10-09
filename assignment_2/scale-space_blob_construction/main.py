#!/usr/bin/env python3

import argparse
import blob
import config
import matplotlib
import os


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--imgpath", "-p", type=str, default=None, help="Image Path", required=True)
    arg = parser.parse_args()
    filepath = arg.imgpath
    _, filename = os.path.split(p=filepath)

    for xf in config.xform_types:
        fig, ax = blob.xform(filepath=filepath, levels=config.levels, xf=xf)
        file_name, file_ext = os.path.splitext(p=filename)
        res_img_dir = os.path.join(config.imgs_res_dir, file_name)
        if not os.path.exists(path=res_img_dir):
            os.mkdir(path=res_img_dir)
        fname = file_name + '_' + xf + file_ext
        res_img_path = os.path.join(res_img_dir, fname)
        matplotlib.pyplot.savefig(fname=res_img_path, dpi=1000, format="jpg", bbox_inches="tight")
        if config.imshow:
            matplotlib.pyplot.show()
    return

if __name__ == "__main__":
    main()