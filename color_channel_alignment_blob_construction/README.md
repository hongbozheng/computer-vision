# Assignment 2 - Fourier-based Alignment and Finding Covariant Neighborhoods
##### Instructor: D.A. Forsyth
Deadline --- Monday, October 9th, 11:59:59 PM

For the first part of this assignment, we will revisit the color channel alignment that you performed in Assignment 1.
The goal in this assignment is to perform color channel alignment using the Fourier transform. As I said in lecture
(and in notes, and on slides), convolution in the spatial domain translates to multiplication in the frequency domain.
Further, the Fast Fourier Transform algorithm computes a transform in O(N M log N M) operations for an N by M image.
As a result, Fourier-based alignment may provide an efficient alternative to sliding window alignment approaches for
high-resolution images.

The goal of Part 2 of the assignment is to build a blob coordinate system as discussed in lecture (and slides, and notes!).

## Developers
* Hongbo Zheng [NetID: hongboz2]