# Gets useful analysis of splits as analysed by given a directory with structure
# dir_name
#   - file1.bvh
#   - file1.wav
#   - file1.json
#   - file2.bvh
#   - file2.wav
#   - file2.json
#   - ...

# these useful stats include:
# number of words per split
# time/split
import numpy as np
from matplotlib import pyplot as plt
from argparse import ArgumentParser

import json
import os
import sys
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)

from caro_tests.bvh_helpers import get_positions, get_low_velocity_hand_points, timestr_to_float, get_times_of_splits
from pydub import AudioSegment
import json


def get_wc_histogram(transcript_files):
    transcripts = []
    for t in transcript_files:
        with open(t) as jt:
            d = json.load(jt)
            try:
                transcripts.append(d['transcript'])
            except:
                print('failed to get transcript for ', t)
    # transcript is now list of lists
    wcs = []
    for t in transcripts:
        wcs.append(len(t))

    bin_size = 10
    bins = np.arange(0, 100, bin_size)
    plt.xlim([0, 90])

    plt.hist(wcs, bins=bins, alpha=0.5)
    plt.title('Word count distribution for split')
    plt.xlabel('words/gesture (bin size = %s)' % bin_size)
    plt.ylabel('count')

    plt.show()


if __name__ == "__main__":
    # Setup parameter parser
    parser = ArgumentParser()
    parser.add_argument('--analyze_dir', '-ad', default="",
                                   help="directory with split files to be analyzed")
    params = parser.parse_args()
    fns = os.listdir(params.analyze_dir)

    fns = [os.path.join(params.analyze_dir, f) for f in fns]

    transcript_files = [f for f in fns if f.endswith('.json')]
    get_wc_histogram(transcript_files)

