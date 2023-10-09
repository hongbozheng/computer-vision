#!/usr/bin/env python3

import align
import config
import cv2
import logger
import matplotlib.pyplot
import numpy
import os
import PIL.Image


def main():
    matplotlib.pyplot.rc(group="font", family="serif")
    # High Resolution Image Alignment
    if config.high_res:
        logger.log_info("Fourier-based color channel alignment with low-resolution images")
        high_res_img_names = os.listdir(path=config.high_res_images_dir)
        if config.LoG_filter:
            if not os.path.exists(path=config.LoG_ft_align_results_dir):
                os.mkdir(path=config.LoG_ft_align_results_dir)
        else:
            if not os.path.exists(path=config.ft_align_results_dir):
                os.mkdir(path=config.ft_align_results_dir)

        for filename in high_res_img_names:
            logger.log_info("Processing file %s" % filename)
            filepath = os.path.join(config.high_res_images_dir, filename)
            mat_bgr = align.align(filepath=filepath, high_res=config.high_res)
            for (i, mat) in enumerate(mat_bgr):
                m = mat[0]
                inv_ft_0 = mat[1]
                inv_ft_1 = mat[2]

                file_name, file_ext = os.path.splitext(p=filename)
                if config.LoG_filter:
                    res_img_dir = file_name + '_' + config.base_ch_order[i]
                    res_img_dir = os.path.join(config.LoG_ft_align_results_dir, res_img_dir)
                else:
                    res_img_dir = file_name + '_' + config.base_ch_order[i]
                    res_img_dir = os.path.join(config.ft_align_results_dir, res_img_dir)
                if not os.path.exists(path=res_img_dir):
                    os.mkdir(path=res_img_dir)
                if i == 0:
                    inv_ft_name_0 = "inv_ft_" + config.base_ch_order[1] + ".jpg"
                    inv_ft_name_1 = "inv_ft_" + config.base_ch_order[2] + ".jpg"
                    if config.LoG_filter:
                        title_0 = ("Inverse Fourier Transform with LoG\n" + "Align " + config.base_ch_order[1] + " to "
                                   + config.base_ch_order[i])
                        title_1 = ("Inverse Fourier Transform with LoG\n" + "Align " + config.base_ch_order[2] + " to "
                                   + config.base_ch_order[i])
                    else:
                        title_0 = ("Inverse Fourier Transform without  LoG\n" + "Align " + config.base_ch_order[1]
                                   + " to " + config.base_ch_order[i])
                        title_1 = ("Inverse Fourier Transform without LoG\n" + "Align " + config.base_ch_order[2]
                                   + " to " + config.base_ch_order[i])
                elif i == 1:
                    inv_ft_name_0 = "inv_ft_" + config.base_ch_order[0] + ".jpg"
                    inv_ft_name_1 = "inv_ft_" + config.base_ch_order[2] + ".jpg"
                    if config.LoG_filter:
                        title_0 = ("Inverse Fourier Transform with LoG\n" + "Align " + config.base_ch_order[0] + " to "
                                   + config.base_ch_order[i])
                        title_1 = ("Inverse Fourier Transform with LoG\n" + "Align " + config.base_ch_order[2] + " to "
                                   + config.base_ch_order[i])
                    else:
                        title_0 = ("Inverse Fourier Transform without  LoG\n" + "Align " + config.base_ch_order[0]
                                   + " to " + config.base_ch_order[i])
                        title_1 = ("Inverse Fourier Transform without LoG\n" + "Align " + config.base_ch_order[2]
                                   + " to " + config.base_ch_order[i])
                elif i == 2:
                    inv_ft_name_0 = "inv_ft_" + config.base_ch_order[0] + ".jpg"
                    inv_ft_name_1 = "inv_ft_" + config.base_ch_order[1] + ".jpg"
                    if config.LoG_filter:
                        title_0 = ("Inverse Fourier Transform with LoG\n" + "Align " + config.base_ch_order[0] + " to "
                                   + config.base_ch_order[i])
                        title_1 = ("Inverse Fourier Transform with LoG\n" + "Align " + config.base_ch_order[1] + " to "
                                   + config.base_ch_order[i])
                    else:
                        title_0 = ("Inverse Fourier Transform without  LoG\n" + "Align " + config.base_ch_order[0]
                                   + " to " + config.base_ch_order[i])
                        title_1 = ("Inverse Fourier Transform without LoG\n" + "Align " + config.base_ch_order[1]
                                   + " to " + config.base_ch_order[i])

                res_img_path = os.path.join(res_img_dir, filename)
                cv2.imwrite(filename=res_img_path, img=m)

                matplotlib.pyplot.figure()
                matplotlib.pyplot.imshow(X=numpy.real(val=inv_ft_0))
                matplotlib.pyplot.title(label=title_0)
                matplotlib.pyplot.colorbar()
                inv_ft_path_0 = os.path.join(res_img_dir, inv_ft_name_0)
                matplotlib.pyplot.savefig(fname=inv_ft_path_0, dpi=1000, format="jpg", bbox_inches="tight")
                matplotlib.pyplot.close()

                matplotlib.pyplot.figure()
                matplotlib.pyplot.imshow(X=numpy.real(val=inv_ft_1))
                matplotlib.pyplot.title(label=title_1)
                matplotlib.pyplot.colorbar()
                inv_ft_path_1 = os.path.join(res_img_dir, inv_ft_name_1)
                matplotlib.pyplot.savefig(fname=inv_ft_path_1, dpi=1000, format="jpg", bbox_inches="tight")
                matplotlib.pyplot.close()

                if config.imshow:
                    cv2.imshow(winname="High-Res Image Alignment", mat=m)
                    cv2.waitKey(0)

    # Low Resolution Image Alignment
    else:
        logger.log_info("Fourier-based color channel alignment with high-resolution images")
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
                    res_img_dir = os.path.join(config.LoG_ft_align_results_dir, res_img_dir)
                else:
                    res_img_dir = file_name + '_' + config.base_ch_order[i]
                    res_img_dir = os.path.join(config.ft_align_results_dir, res_img_dir)
                if not os.path.exists(path=res_img_dir):
                    os.mkdir(path=res_img_dir)
                if i == 0:
                    inv_ft_name_0 = "inv_ft_" + config.base_ch_order[1] + ".jpg"
                    inv_ft_name_1 = "inv_ft_" + config.base_ch_order[2] + ".jpg"
                    if config.LoG_filter:
                        title_0 = ("Inverse Fourier Transform with LoG\n" + "Align " + config.base_ch_order[1] + " to "
                                   + config.base_ch_order[i])
                        title_1 = ("Inverse Fourier Transform with LoG\n" + "Align " + config.base_ch_order[2] + " to "
                                   + config.base_ch_order[i])
                    else:
                        title_0 = ("Inverse Fourier Transform without  LoG\n" + "Align " + config.base_ch_order[1]
                                   + " to " + config.base_ch_order[i])
                        title_1 = ("Inverse Fourier Transform without LoG\n" + "Align " + config.base_ch_order[2]
                                   + " to " + config.base_ch_order[i])
                elif i == 1:
                    inv_ft_name_0 = "inv_ft_" + config.base_ch_order[0] + ".jpg"
                    inv_ft_name_1 = "inv_ft_" + config.base_ch_order[2] + ".jpg"
                    if config.LoG_filter:
                        title_0 = ("Inverse Fourier Transform with LoG\n" + "Align " + config.base_ch_order[0] + " to "
                                   + config.base_ch_order[i])
                        title_1 = ("Inverse Fourier Transform with LoG\n" + "Align " + config.base_ch_order[2] + " to "
                                   + config.base_ch_order[i])
                    else:
                        title_0 = ("Inverse Fourier Transform without  LoG\n" + "Align " + config.base_ch_order[0]
                                   + " to " + config.base_ch_order[i])
                        title_1 = ("Inverse Fourier Transform without LoG\n" + "Align " + config.base_ch_order[2]
                                   + " to " + config.base_ch_order[i])
                elif i == 2:
                    inv_ft_name_0 = "inv_ft_" + config.base_ch_order[0] + ".jpg"
                    inv_ft_name_1 = "inv_ft_" + config.base_ch_order[1] + ".jpg"
                    if config.LoG_filter:
                        title_0 = ("Inverse Fourier Transform with LoG\n" + "Align " + config.base_ch_order[0] + " to "
                                   + config.base_ch_order[i])
                        title_1 = ("Inverse Fourier Transform with LoG\n" + "Align " + config.base_ch_order[1] + " to "
                                   + config.base_ch_order[i])
                    else:
                        title_0 = ("Inverse Fourier Transform without  LoG\n" + "Align " + config.base_ch_order[0]
                                   + " to " + config.base_ch_order[i])
                        title_1 = ("Inverse Fourier Transform without LoG\n" + "Align " + config.base_ch_order[1]
                                   + " to " + config.base_ch_order[i])

                image = PIL.Image.fromarray(obj=m)
                res_img_path = os.path.join(res_img_dir, filename)
                image.save(fp=res_img_path)

                matplotlib.pyplot.figure()
                matplotlib.pyplot.imshow(X=numpy.real(val=inv_ft_0))
                matplotlib.pyplot.title(label=title_0)
                matplotlib.pyplot.colorbar()
                inv_ft_path_0 = os.path.join(res_img_dir, inv_ft_name_0)
                matplotlib.pyplot.savefig(fname=inv_ft_path_0, dpi=1000, format="jpg", bbox_inches="tight")
                matplotlib.pyplot.close()

                matplotlib.pyplot.figure()
                matplotlib.pyplot.imshow(X=numpy.real(val=inv_ft_1))
                matplotlib.pyplot.title(label=title_1)
                matplotlib.pyplot.colorbar()
                inv_ft_path_1 = os.path.join(res_img_dir, inv_ft_name_1)
                matplotlib.pyplot.savefig(fname=inv_ft_path_1,  dpi=1000, format="jpg", bbox_inches="tight")
                matplotlib.pyplot.close()

                if config.imshow:
                    image.show(title="Low-Red Image Alignment")
                    matplotlib.pyplot.show()

    return


if __name__ == "__main__":
    main()