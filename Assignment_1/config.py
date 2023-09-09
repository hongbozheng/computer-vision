import logger

basic_alignment_image_paths = ["data/00125v.jpg", "data/00149v.jpg", "data/00153v.jpg",
                               "data/00351v.jpg", "data/00398v.jpg", "data/01112v.jpg"]
multiscale_alignment_image_paths = ["data_hires/01047u.tif", "data_hires/01657u.tif", "data_hires/01861a.tif"]
metric = "SSD"
basic_alignment_results_dir = "basic_alignment_results"
multiscale_alignment_results_dir = "multiscale_alignment_results"
imshow = False
log_level = logger.LogLevel.info