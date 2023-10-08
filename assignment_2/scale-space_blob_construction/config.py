import logger

# directories
images_dir = "images"
images_result_dir = "results"

# good feature to track
# maxCorners   - Maximum number of corners to return. If there are more corners than are found, the strongest of them is
#                returned. if <= 0 implies that no limit on the maximum is set and all detected corners are returned
# qualityLevel - Parameter characterizing the minimal accepted quality of image corners. See the above paragraph for
#                explanation
# minDistance  - Minimum possible Euclidean distance between the returned corners
# mask         - Optional region of interest. If the image is not empty it specifies the region in which the corners are
#                detected
# blockSize    - Size of an average block for computing a derivative covariation matrix over each pixel neighborhood
# useHarrisDetector - whether to use Shi-Tomasi or Harris Corner
# k                 - Free parameter of the Harris detector
max_corners = 0
quality_level = 0.10
min_dist = 1
blk_size = 10
harris = True
k = 0.05
ksize = 5

# mark corners & plt blobs & orientations
mark_size = 8
arrow_head_w = 3
arrow_head_l = 2
mark_color = "orange"
blob_color = "cyan"
arrow_color = "magenta"

# test
levels = 15
imshow = False
log_level = logger.LogLevel.info