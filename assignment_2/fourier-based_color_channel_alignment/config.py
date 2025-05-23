import logger

# directories
low_res_images_dir = "data"
high_res_images_dir = "data_hires"
LoG_ft_align_results_dir = "LoG_fourier-based_alignment_results"
ft_align_results_dir = "fourier-based_alignment_results"

# remove border & split images
low_res_border_search_range = 35
high_res_border_search_range = 250
white_threshold = 245
black_threshold = 35

# gaussian blur
gaussian_blur_kernel_size = (3, 3)
gaussian_blur_sigmaX = 1
gaussian_blur_sigmaY = 1

# laplacian
laplacian_kernel_size = 3

# test
base_ch_order = ['B', 'G', 'R']
high_res = False
LoG_filter = False
imshow = False
log_level = logger.LogLevel.INFO
