import logger

single_scale_alignment_images_dir = "data"
multiscale_alignment_image_paths = ["data_hires/01047u.tif", "data_hires/01657u.tif", "data_hires/01861a.tif"]
single_scale_alignment_border_width_threshold = 35
multiscale_alignment_border_width_threshold = 175
white_threshold = 245
black_threshold = 35
metric = "NCC"
anti_aliasing_sigma = 1
single_scale_alignment_results_dir = "single_scale_alignment_results"
multiscale_alignment_results_dir = "multiscale_alignment_results"
imshow = False
log_level = logger.LogLevel.info