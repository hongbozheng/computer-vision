import logger

single_scale_alignment_images_dir = "data"
multiscale_alignment_images_dir = "data_hires"
single_scale_alignment_border_search_range = 35
multiscale_alignment_border_search_range = 250
white_threshold = 245
black_threshold = 35
metric = "SSD"
displacement_range = 35
single_scale_align = False
single_scale_alignment_results_dir = "single_scale_alignment_results"
multiscale_alignment_results_dir = "multiscale_alignment_results"
imshow = False
log_level = logger.LogLevel.info