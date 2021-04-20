from caro_tests.bvh_helpers import *
from local_modules.pymo.viz_tools import save_fig, draw_stickfigure, draw_stickfigure3d
import os
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# given directory with a buncha bvh files,
# resample them to get to frame rate 0.05
import numpy as np
import pandas as pd

from argparse import ArgumentParser
from tqdm import tqdm
import math
import os
import sys
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)

RESAMPLE_FR = 0.05
RESAMPLE_FPS = 20
ORIG_FR = 0.0166667
ORIG_FPS = 60


# example usage:
# python -m caro_tests.resample_bvh --resample_dir Splits/NaturalTalking_008-sentence/

def resample_bvh(bvh_file, output_name):
    f = open(bvh_file, "r")
    resampled = open(output_name, 'w')

    # copy hierarchy
    line = ""
    while 'MOTION' not in str(line):
        line = f.readline()
        resampled.write(str(line))

    line = str(f.readline())
    orig_fc = float(line.split(' ')[1])
    new_fc = math.floor(orig_fc / (ORIG_FPS / RESAMPLE_FPS))        # functionally dividing by 3.
    resampled.write(str('Frames: %s\n' % new_fc))
    resampled.write(str('Frame Time: %s\n' % RESAMPLE_FR))

    frames_added = 0
    f.readline()                    # skip past the Frame Time of the original file
    line = str(f.readline())
    while line != "":
        resampled.write(str(line))
        frames_added += 1
        if frames_added == new_fc:      # don't want to go over the initial frame count
            break
        f.readline()
        f.readline()
        line = str(f.readline())    # we can do this as many times as we want and it'll just return ''!

    f.close()
    resampled.close()


if __name__ == "__main__":
    # Setup parameter parser
    parser = ArgumentParser()
    parser.add_argument('--resample_dir', '-dir', default="",
                                   help="directory to go through bvh files from")
    parser.add_argument('--file', '-f', default="",
                                   help="file to resample.")

    params = parser.parse_args()

    if params.file:
        resample_bvh(params.file)
    else:
        fns = os.listdir(params.resample_dir)
        fns = [os.path.join(params.resample_dir, f) for f in fns]
        bvh_files = [f for f in fns if f.endswith('.bvh')]
        for f in tqdm(bvh_files):
            print(f)
            output_file = f.split('.')[0] + '_resampled.bvh'
            resample_bvh(f, output_file)
