import logger

# directories
single_scale_alignment_images_dir = "data"
multiscale_alignment_images_dir = "data_hires"
single_scale_alignment_results_dir = "single_scale_alignment_results"
multiscale_alignment_results_dir = "multiscale_alignment_results"

# remove border & split images
single_scale_alignment_border_search_range = 35
multiscale_alignment_border_search_range = 250
white_threshold = 245
black_threshold = 35

# find displacement
metric = "NCC"
displacement_range = 50

# pyramid find displacement
gaussian_blur_kernel_size = (3, 3)
gaussian_blur_sigmaX = 1
gaussian_blur_sigmaY = 1
num_pyramid_levels = 5

# test
img_pyr = True
imshow = False
log_level = logger.LogLevel.info