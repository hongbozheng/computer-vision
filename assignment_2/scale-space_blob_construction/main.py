#!/usr/bin/env python3

import blob
import config
import os


def main():
    img_names = os.listdir(path=config.images_dir)

    for filename in img_names:
        filepath = os.path.join(config.images_dir, filename)
        blob.construct_blob(filepath=filepath, levels=config.levels)
    return

if __name__ == "__main__":
    main()