import logger

# directories
img_dir = "croppedyale"
res_dir = "results"

# filter image
pixel_val_thres = 50
dark_pixel_ratio_thres = 0.75

# height map
integration_methods = {"average", "column", "row", "random"}

# test
num_paths = 35
imshow = False
log_level = logger.LogLevel.INFO
