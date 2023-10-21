import logger

# directories
img_dir = "images"
res_dir = "stitched_results"

dist_thres = 1.0e4
ransac_num_iters = 1000
ransac_thres = 1.0

# test
multi_imgs = False
plt_inliner_matches = True
imshow = True
log_level = logger.LogLevel.info