import numpy as np
import pandas as pd

from argparse import ArgumentParser

import glob
import os
import sys
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)

from caro_tests.bvh_helpers import get_positions, get_low_velocity_hand_points, timestr_to_float, get_times_of_splits
from pydub import AudioSegment
import json

import joblib as jl

## the goal of this file is to identify where to cut the gestures
## the criteria for that include:
## when both hands have a velocity below a certain threshold
## when both hands are in the rest position?

# usage
# python bvh_splitter.py --raw_dir <path/to/raw> --file_name <i.e. NaturalTalking_001> --dest_dir <Splits>


# takes list of frames to split bvh file at
# saves as many files as splits+1
def split_bvh_at_frames(orig_bvh, bvh_name, frame_splits):
    f = open(orig_bvh, "r")

    # create files for split
    splits = []
    prev_frame = 0
    for i in range(0, len(frame_splits)+1):
        if i == len(frame_splits):
            fn = f'{bvh_name}_split_{i}_frame_{prev_frame}_+.bvh'
            splits.append(open(fn, 'wt'))
        else:
            fn = f'{bvh_name}_split_{i}_frame_{prev_frame}_{frame_splits[i]}.bvh'
            splits.append(open(fn, 'wt'))
            prev_frame = frame_splits[i]
        # print(fn)

    # copy hierarchy
    line = ""
    while 'MOTION' not in str(line):
        line = f.readline()
        for s in splits:
            s.write(str(line))

    # get the frame counts for each split
    framecounts = []
    line = str(f.readline())
    orig_framecount = int(line.split("\t")[1])
    for i in range(0, len(frame_splits)+1):
        if i == 0:
            framecounts.append(frame_splits[i])
        elif i < len(frame_splits):
            framecounts.append(frame_splits[i] - frame_splits[i-1])
        else:
            framecounts.append(orig_framecount - frame_splits[-1])
    for i in range(len(splits)):
        splits[i].write(str('Frames: %s\n' % framecounts[i]))

    # add the frame time
    line = f.readline()
    for s in splits:
        s.write(str(line))

    # add frames as we go
    cur_framecount = 0
    i = 0
    while i < len(splits):
        if i == len(frame_splits):
            while f:
                line = f.readline()
                splits[i].write(str(line))
                if line == "":
                    break
        else:
            while cur_framecount < frame_splits[i]:
                line = f.readline()
                splits[i].write(str(line))
                cur_framecount += 1
        i += 1

    f.close()
    for s in splits:
        s.close()


# times is in seconds
def split_audio_at_times(wav_f, fn, times):
    orig_audio = AudioSegment.from_wav(wav_f)
    for i in range(0, len(times)):
        if i == 0:
            wav_name = f'{fn}_split_{i}_time_{0}_{times[i]}.wav'
            newAudio = orig_audio[0:(times[i] * 1000)]      # works in ms
            newAudio.export(wav_name, format="wav")  # Exports to a wav file in the current path
        else:
            wav_name = f'{fn}_split_{i}_time_{times[i-1]}_{times[i]}.wav'
            newAudio = orig_audio[(1000* times[i-1]):(1000* times[i])]
            newAudio.export(wav_name, format="wav")  # Exports to a wav file in the current path


def split_transcript_at_times(transcript_f, fn, times):
    with open(transcript_f) as f:
        data = json.load(f)
        all_words = []
        for a in data:
            all_words += a['alternatives'][0]['words']
        i = 0
        j = 0
        curr_words = []
        transcript = []
        while i < len(all_words):
            if timestr_to_float(all_words[i]['start_time']) < times[j]:   # keep going on this transcript
                curr_words.append(all_words[i])
                transcript.append(all_words[i]['word'])
            elif j == 0:       # time to print a new transcript
                trans = {
                    'transcript': transcript,
                    'words': curr_words
                }
                t_name = f'{fn}_split_{j}_time_{0}_{times[j]}.json'
                with open(t_name, 'w') as out:
                    json.dump(trans, out)
                j += 1
                curr_words = [all_words[i]]
                transcript = [all_words[i]['word']]
            else:
                trans = {
                    'transcript': transcript,
                    'words': curr_words
                }
                t_name = f'{fn}_split_{j}_time_{times[j-1]}_{times[j]}.json'
                with open(t_name, 'w') as out:
                    json.dump(trans, out)
                j += 1
                curr_words = [all_words[i]]
                transcript = [all_words[i]['word']]
            i += 1

        # get the last one
        t_name = f'{fn}_split_{j}_frame_{times[j]}_+.json'
        with open(t_name, 'w') as out:
            json.dump(curr_words, out)


if __name__ == "__main__":
    # Setup parameter parser
    parser = ArgumentParser()
    parser.add_argument('--raw_dir', '-orig', default="../dataset/raw_data/",
                                   help="Path where original motion files (in BVH format) are stored")
    parser.add_argument('--dest_dir', '-dest', default="../../dataset/raw_data/Motion",
                                   help="Path where extracted motion features will be stored")
    parser.add_argument('--pipeline_dir', '-pipe', default="../utils/",
                        help="Path where the motion data processing pipeline will be stored")
    parser.add_argument('--file_name', '-bvh', default="NaturalTalking_010",
                        help="bvh file to extract")



    params = parser.parse_args()
    bvh_name = os.path.join(params.raw_dir, 'Motion', params.file_name + '.bvh')
    wav_name = os.path.join(params.raw_dir, 'Audio', params.file_name + '.wav')
    txt_name = os.path.join(params.raw_dir, 'Transcripts', params.file_name + '.json')

    if not os.path.exists(params.dest_dir):
        os.mkdir(params.dest_dir)

    print('getting motion data')
    modat = get_positions(bvh_name)[0]      ## the 0 here is because we only operate on 1 file at a time

    lv = get_low_velocity_hand_points(modat)

    dest_file = os.path.join(params.dest_dir, params.file_name)

    # create the bvh file splits at the points of low velocity
    split_bvh_at_frames(bvh_name, dest_file, lv)

    # now split audio and text
    times = get_times_of_splits(lv)
    split_audio_at_times(wav_name, dest_file, times)
    split_transcript_at_times(txt_name, dest_file, times)