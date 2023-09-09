#!/usr/bin/env python3

import align


def main():
    # Single-Scale Alignment
    align.single_scale_align()

    # Multiscale Alignment
    # align.multiscale_align(image=config.multiscale_alignment_image_paths[0])

    return


if __name__ == "__main__":
    main()