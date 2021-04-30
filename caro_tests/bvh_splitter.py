import numpy as np
import pandas as pd

from argparse import ArgumentParser

import math
import os
import sys
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)

from caro_tests.bvh_helpers import get_positions, get_low_velocity_hand_points, timestr_to_float, \
                                   get_times_of_splits, get_frames_of_splits
from pydub import AudioSegment
import json

## example usage
# python -m caro_tests.bvh_splitter --raw_dir ../raw_data --dest_dir Splits/NaturalTalking_008-tconst10 --file_name NaturalTalking_008 --split_by timeconst -tc 10


## the goal of this file is to identify where to cut the gestures
## the criteria for that include:
## when both hands have a velocity below a certain threshold
## when both hands are in the rest position?

# usage
# python bvh_splitter.py --raw_dir <path/to/raw> --file_name <i.e. NaturalTalking_001> --dest_dir <Splits>


# takes list of frames to split bvh file at
# saves as many files as splits+1
# if downsampling as well, samples from 0.0166 fps to 0.05 fps
# so only take every 3 frames...?
TIME_CONST = 10      # number of seconds to make gesture splits doing a timeconst
WORD_CONST = 10      # number of words to make gesture splits doing a wordconst


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
            if line != chr(10): # skip newlines????
                l = str(line)
                s.write(str(line))

    # get the frame counts for each split
    framecounts = []
    line = str(f.readline())
    if line == chr(10):  # skip newlines????
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
        fc = framecounts[i]
        splits[i].write(str('Frames: %s\n' % fc))

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

    wav_name = f'{fn}_split_{i+1}_time_{times[-1]}_+.wav'
    newAudio = orig_audio[(1000 * times[i]):]
    newAudio.export(wav_name, format="wav")  # Exports to a wav file in the current path


## TODO: MAJORLY!!!! DOESN'T CURRENTLY GET TRANSCRIPT FOR LAST FILE!!!!
def split_transcript_at_times(transcript_f, fn, times):
    with open(transcript_f) as f:
        data = json.load(f)
        all_words = []
        for a in data:
            all_words += a['alternatives'][0]['words']
        words_iter = 0
        times_iter = 0
        curr_words = []
        transcript = []
        while words_iter < len(all_words) and times_iter < len(times):
            if timestr_to_float(all_words[words_iter]['start_time']) < times[times_iter]:   # keep going on this transcript
                curr_words.append(all_words[words_iter])
                transcript.append(all_words[words_iter]['word'])
            elif times_iter == 0:       # time to print a new transcript
                trans = {
                    'transcript': transcript,
                    'words': curr_words
                }
                t_name = f'{fn}_split_{times_iter}_time_{0}_{times[times_iter]}.json'
                with open(t_name, 'w') as out:
                    json.dump(trans, out)
                times_iter += 1
                curr_words = [all_words[words_iter]]
                transcript = [all_words[words_iter]['word']]
            else:
                trans = {
                    'transcript': transcript,
                    'words': curr_words
                }
                t_name = f'{fn}_split_{times_iter}_time_{times[times_iter-1]}_{times[times_iter]}.json'
                with open(t_name, 'w') as out:
                    json.dump(trans, out)
                times_iter += 1
                curr_words = [all_words[words_iter]]
                transcript = [all_words[words_iter]['word']]
            words_iter += 1

        # get the last one
        t_name = f'{fn}_split_{times_iter+1}_time_{times[-1]}_+.json'
        with open(t_name, 'w') as out:
            trans = {
                'transcript': transcript,
                'words': curr_words
            }
            json.dump(trans, out)


def get_sentence_ending_times(transcript):
    sentence_end_times = []
    with open(transcript) as t:
        data = json.load(t)
        all_words = []
        for a in data:
            all_words += a['alternatives'][0]['words']
        for w in all_words:
            if '.' in w['word']:
                t = float(w['end_time'].split('s')[0])  # annoying formatting
                sentence_end_times.append(t + 1)  # just add a buffer second.
    return sentence_end_times


# split gestures into chunks containing wc number of words
def get_wordcount_ending_times(transcript, wc):
    end_times = []
    with open(transcript) as t:
        data = json.load(t)
        all_words = []
        for a in data:
            all_words += a['alternatives'][0]['words']

        i = wc
        while i < len(all_words):
            t = float(all_words[i]['end_time'].split('s')[0])  # annoying formatting
            end_times.append(t + 1)     # 1 second buffer why  not
            i += wc

    return end_times


# split gestures into chunks of length time_interval
def get_timeconst_ending_times(transcript, time_interval):
    times = []
    with open(transcript) as t:
        data = json.load(t)
        last_time = int(float(data[-1]['alternatives'][0]['words'][-1]['end_time'].split('s')[0]))

        for i in range(0, last_time, time_interval):
            times.append(i)

    return times


def split_bvh_files(bvh_name, txt_name, wav_name, dest_dir, output_name=None, split_by=None, wordcount=10, timeconst=10):
    split_frames = []
    split_times = []
    if split_by == 'low_velocity':
        print('getting motion data')
        modat = get_positions(bvh_name)[0]  ## the 0 here is because we only operate on 1 file at a time
        split_frames = get_low_velocity_hand_points(modat)
        split_times = get_times_of_splits(split_frames)
    elif split_by == 'sentence':
        split_times = get_sentence_ending_times(txt_name)
        split_frames = get_frames_of_splits(split_times)
    elif split_by == 'wordcount':
        split_times = get_wordcount_ending_times(txt_name, int(wordcount))
        split_frames = get_frames_of_splits(split_times)
    elif split_by == 'timeconst':
        split_times = get_timeconst_ending_times(txt_name, int(timeconst))
        split_frames = get_frames_of_splits(split_times)
    elif split_by == 'fillers':
        split_times = get_filler_word_times(txt_name)
        split_frames = get_frames_of_splits(split_times)

    print('SPLIT FRAMES: ', split_frames)
    print('SPLIT TIMES:', split_times)

    dest_file = os.path.join(dest_dir, output_name)

    # create the bvh file splits at the points of low velocity
    print('splitting: ', bvh_name)
    split_bvh_at_frames(bvh_name, dest_file, split_frames)
    # now split audio and text
    split_audio_at_times(wav_name, dest_file, split_times)
    split_transcript_at_times(txt_name, dest_file, split_times)


def get_full_transcript(transcript):
    with open(transcript) as t:
        data = json.load(t)
        all_words = []
        for a in data:
            all_words += a['alternatives'][0]['words']

        txt = ""
        for w in all_words:
            txt += w['word'] + ' '
        return all_words, txt


def get_filler_word_times(transcript):
    with open(transcript) as t:
        data = json.load(t)
        all_words = []
        for a in data:
            all_words += a['alternatives'][0]['words']

        times = []
        for w in all_words:
            if w['word'] in ['like', 'eh'] or '.' in w['word']:
                times.append(float(w['end_time'].split('s')[0]))

        return times


# TODO: be clever about this, but try to split on the root times.
# tricky to align the transcript to word again.
def get_dependency_root_times(transcript):
    all_words, txt = get_full_transcript(transcript)

    root_ids = []
    import spacy
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(txt)


if __name__ == "__main__":
    # Setup parameter parser
    parser = ArgumentParser()
    parser.add_argument('--raw_dir', '-orig', default="../dataset/raw_data/",
                                   help="Path where original motion files (in BVH format) are stored")
    parser.add_argument('--dest_dir', '-dest', default="../../dataset/raw_data/tmp",
                                   help="Path where extracted motion features will be stored")
    parser.add_argument('--file_name', '-bvh', default="NaturalTalking_010",
                                   help="bvh file to extract")
    parser.add_argument('--split_by', '-sp', default="low_velocity",
                                   help="how to split up gestures <low_velocity, sentence, wordcount, timeconst>")
    parser.add_argument('--wordcount', '-wc', default=10,
                                   help="if split gestures by wordcount, how many words")
    parser.add_argument('--timeconst', '-tc', default=10,
                                   help="if split gestures by time, how many seconds")


    params = parser.parse_args()
    bvh_name = os.path.join(params.raw_dir, 'Motion', params.file_name + '.bvh')
    wav_name = os.path.join(params.raw_dir, 'Audio', params.file_name + '.wav')
    txt_name = os.path.join(params.raw_dir, 'Transcripts', params.file_name + '.json')

    dest_dir = params.dest_dir
    if not os.path.exists(dest_dir):
        os.mkdir(dest_dir)

    split_bvh_files(bvh_name, txt_name, wav_name, dest_dir, output_name=params.file_name, split_by=params.split_by)