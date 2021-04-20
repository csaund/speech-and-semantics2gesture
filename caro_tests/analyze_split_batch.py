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
import math
from caro_tests.bvh_helpers import sorter
from pydub import AudioSegment
import json


def get_wc_histogram(transcript_files, od=None):
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

    save = None
    if od:
        save = os.path.join(od, 'wc_hist.png')

    show_hist(wcs, bin_size=10, x_lim=[0, 90], title='Word count distribution for split', save_to=save)


# haha use naming convention like an absolute chump
# TODO but actually maybe actually use the files.
def get_time_histogram(audio_files, od=None):
    times = []
    for f in audio_files:
        fn = (f.split('_'))
        i = 0
        while i < len(fn) and fn[i] != 'time':      # this is all a bit of a hack because fns aren't the same.
            i += 1
        if i < len(fn):
            times.append(fn[i+1])
    # TODO skips last one, but since we're just getting a rough idea that's probably ok.
    # print(times)
    ls = []
    for i in range(len(times)-1):
        ls.append(float(times[i+1]) - float(times[i]))

    save = None
    if od:
        save = os.path.join(od, 'time_hist.png')

    show_hist(ls, bin_size=10, x_lim=[0, max(ls)+10], title='Time (seconds) distribution for split', save_to=save)


def show_hist(data, bin_size=10, x_lim=[0, 90], title='', save_to=None):
    txt = "; data: " + str([math.floor(d) for d in data])

    bins = np.arange(0, max(data), bin_size)
    plt.xlim(x_lim)
    plt.hist(data, bins=bins, alpha=0.5)
    plt.title(title + txt, wrap=True)
    plt.xlabel('X/gesture (bin size = %s)' % bin_size)
    plt.ylabel('count')
    # plt.text(0, -3, txt, fontsize=10)

    if save_to:
        plt.savefig(save_to)
    else:
        plt.show()


if __name__ == "__main__":
    # Setup parameter parser
    parser = ArgumentParser()
    parser.add_argument('--analyze_dir', '-ad', default="",
                                   help="directory with split files to be analyzed")
    parser.add_argument('--output_dir', '-od', default=None,
                                   help="directory to put plots in")
    params = parser.parse_args()
    fns = os.listdir(params.analyze_dir)

    fns = [os.path.join(params.analyze_dir, f) for f in fns]
    transcript_files = [f for f in fns if f.endswith('.json')]
    transcript_files.sort(key=sorter)
    get_wc_histogram(transcript_files, params.output_dir)

    audio_files = [f for f in fns if f.endswith('.wav')]
    get_time_histogram(transcript_files, params.output_dir)

