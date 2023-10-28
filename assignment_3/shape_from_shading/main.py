#!/usr/bin/env python3

import config
import logger
import numpy
import PIL.Image
import os
import shape

def main():
    dirs = os.listdir(path=config.img_dir)

    for dir in dirs:
        logger.log_info("Start processing %s" % dir)
        class_dir = os.path.join(config.img_dir, dir)
        albedo_image, surface_normals = shape.recover_surface(class_dir=class_dir)

        res_path = os.path.join(config.res_dir, dir)
        os.makedirs(name=res_path, exist_ok=True)
        im = PIL.Image.fromarray(obj=(albedo_image * 255).astype(numpy.uint8))
        res_img_path = os.path.join(res_path, "albedo.jpg")
        im.save(fp=res_img_path)
        im = PIL.Image.fromarray(obj=(surface_normals[:, :, 0] * 128 + 128).astype(numpy.uint8))
        res_img_path = os.path.join(res_path, "normals_x.jpg")
        im.save(fp=res_img_path)
        im = PIL.Image.fromarray(obj=(surface_normals[:, :, 1] * 128 + 128).astype(numpy.uint8))
        res_img_path = os.path.join(res_path, "normals_y.jpg")
        im.save(fp=res_img_path)
        im = PIL.Image.fromarray(obj=(surface_normals[:, :, 2] * 128 + 128).astype(numpy.uint8))
        res_img_path = os.path.join(res_path, "normals_z.jpg")
        im.save(fp=res_img_path)
        break

    return


if __name__ == "__main__":
    main()