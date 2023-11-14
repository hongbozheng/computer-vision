# Part 2: Fundamental Matrix Estimation, Camera Calibration, Triangulation
# Fundamental Matrix Estimation
import matplotlib.pyplot as plt
import numpy
import PIL.Image

##
## load images and match files for the first example
##

I1 = PIL.Image.open(fp="MP3_part2_data/library1.jpg")
I2 = PIL.Image.open(fp="MP3_part2_data/library2.jpg")
matches = numpy.loadtxt(fname="MP3_part2_data/library_matches.txt")

# this is a N x 4 file where the first two numbers of each row
# are coordinates of corners in the first image and the last two
# are coordinates of corresponding corners in the second image: 
# matches(i,1:2) is a point in the first image
# matches(i,3:4) is a corresponding point in the second image

N = len(matches)

##
## display two images side-by-side with matches
## this code is to help you visualize the matches, you don't need
## to use it to produce the results for the assignment
##

I3 = numpy.zeros(shape=(I1.size[1], I1.size[0]*2, 3))
I3[:, :I1.size[0], :] = I1
I3[:, I1.size[0]:, :] = I2
fig, ax = plt.subplots()
ax.set_aspect("equal")
ax.imshow(numpy.array(I3).astype(dtype=float))
ax.plot(matches[:, 0], matches[:, 1], '+r')
ax.plot(matches[:, 2]+I1.size[0], matches[:, 3], '+r')
ax.plot([matches[:, 0], matches[:, 2]+I1.size[0]], [matches[:, 1], matches[:, 3]], 'r')
plt.show()

##
## display second image with epipolar lines reprojected 
## from the first image
##

# first, fit fundamental matrix to the matches
F = fit_fundamental(matches) # this is a function that you should write
M = numpy.c_[matches[:, 0:2], numpy.ones(shape=(N, 1))].transpose()
L1 = numpy.matmul(F, M).transpose() # transform points from
# the first image to get epipolar lines in the second image

# find points on epipolar lines L closest to matches(:,3:4)
l = numpy.sqrt(L1[:, 0]**2 + L1[:, 1]**2)
L = numpy.divide(L1, numpy.kron(a=numpy.ones((3, 1)), b=l).transpose()) # rescale the line
pt_line_dist = numpy.multiply(L, numpy.c_[matches[:, 2:4], numpy.ones(shape=(N, 1))]).sum(axis=1)
closest_pt = matches[:, 2:4] - numpy.multiply(L[:, 0:2], numpy.kron(a=numpy.ones(shape=(2, 1)), b=pt_line_dist).transpose())

# find endpoints of segment on epipolar line (for display purposes)
pt1 = closest_pt - numpy.c_[L[:, 1], -L[:, 0]]*10 # offset from the closest point is 10 pixels
pt2 = closest_pt + numpy.c_[L[:, 1], -L[:, 0]]*10

# display points and segments of corresponding epipolar lines
fig, ax = plt.subplots()
ax.set_aspect('equal')
ax.imshow(numpy.array(object=I2).astype(dtype=float))
ax.plot(matches[:, 2], matches[:, 3], '+r')
ax.plot([matches[:, 2], closest_pt[:, 0]], [matches[:, 3], closest_pt[:, 1]], 'r')
ax.plot([pt1[:, 0], pt2[:, 0]], [pt1[:, 1], pt2[:, 1]], 'g')
plt.show()


## Camera Calibration

def evaluate_points(M, points_2d, points_3d):
    """
    Visualize the actual 2D points and the projected 2D points calculated from
    the projection matrix
    You do not need to modify anything in this function, although you can if you
    want to
    :param M: projection matrix 3 x 4
    :param points_2d: 2D points N x 2
    :param points_3d: 3D points N x 3
    :return:
    """
    N = len(points_3d)
    points_3d = numpy.hstack(tup=(points_3d, numpy.ones(shape=(N, 1))))
    points_3d_proj = numpy.dot(a=M, b=points_3d.T).T
    u = points_3d_proj[:, 0] / points_3d_proj[:, 2]
    v = points_3d_proj[:, 1] / points_3d_proj[:, 2]
    residual = numpy.sum(a=numpy.hypot(u-points_2d[:, 0], v-points_2d[:, 1]))
    points_3d_proj = numpy.hstack(tup=(u[:, numpy.newaxis], v[:, numpy.newaxis]))
    return points_3d_proj, residual


## Camera Centers


## Triangulation
