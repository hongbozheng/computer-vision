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
        res_img_path = os.path.join(config.imgs_res_dir, filename)
        matplotlib.pyplot.savefig(fname=res_img_path, dpi=1000, format="jpg", bbox_inches="tight")
        if config.imshow:
            matplotlib.pyplot.show()
        break
    return

if __name__ == "__main__":
    main()