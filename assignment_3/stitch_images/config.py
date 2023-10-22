import logger

# directories
img_dir = "images"
img_multi_dir = "images_multi"
res_dir = "stitched_results"

dist_thres = 1.0e4
ransac_num_iters = 1000
ransac_thres = 1.0

# test
multi_imgs = False
plt_inliner_matches = False
imshow = False
log_level = logger.LogLevel.info