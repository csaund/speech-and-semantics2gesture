from bvh_helpers import *
from local_modules.pymo.viz_tools import save_fig, draw_stickfigure, draw_stickfigure3d
import os
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# given directory with a buncha bvh files,
# resample them to get to frame rate 0.05
import numpy as np
import pandas as pd

from argparse import ArgumentParser

import math
import os
import sys
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)

from caro_tests.bvh_helpers import get_positions, get_low_velocity_hand_points, timestr_to_float, get_times_of_splits
from pydub import AudioSegment
import json

import joblib as jl


def resample_bvh(bvh_file):
    return


if __name__ == "__main__":
    # Setup parameter parser
    parser = ArgumentParser()
    parser.add_argument('--resample_dir', '-dir', default="",
                                   help="directory to go through bvh files from")

    params = parser.parse_args()
    fns = os.listdir(params.resample_dir)

    fns = [os.path.join(params.resample_dir, f) for f in fns]
    bvh_files = [f for f in fns if f.endswith('.bvh')]
    for f in bvh_files:
        resample_bvh(f)