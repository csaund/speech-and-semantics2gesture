import numpy as np
import pandas as pd

from argparse import ArgumentParser

import math
import os
import sys
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)

from caro_tests.bvh_helpers import sorter
from caro_tests.bvh_splitter import split_bvh_files
from caro_tests.collect_transcript_from_gestures import collect_transcript_from_dir
from caro_tests.analyze_split_batch import get_wc_histogram, get_time_histogram
from caro_tests.mirror_bvh import mirror_downsample_bvh
from caro_tests.mp4_wav_to_mov import combine_av_for_dir
from caro_tests.bvh_to_mp4_batch import bvh_to_video
from caro_tests.get_bert_embeddings import get_embeddings_for_dir
from tqdm import tqdm
from pathlib import Path


# from all the other helper scripts, takes a raw data file and produces a folder with
# split bvhs, wavs, transcripts, and mp4s of the gestures as they are split.
# example usage:
# python -m caro_tests.process_data --raw_dir ../raw_data/ --file_name NaturalTalking_002 --split_by sentence --dest_dir Splits/NaturalTalking_002-sentence/

LOCAL_SERVER_URL = "http://localhost:5001"


if __name__ == "__main__":
    # Setup parameter parser
    parser = ArgumentParser()
    parser.add_argument('--raw_dir', '-orig', default="../dataset/raw_data/",
                                   help="Path where original motion files (in BVH format) are stored")
    parser.add_argument('--dest_dir', '-dest', default="../../dataset/raw_data/tmp",
                                   help="Path where extracted motion features will be stored")
    parser.add_argument('--file_name', '-bvh', default="",
                                   help="bvh file to extract e.g. NaturalTalking_008")
    parser.add_argument('--split_by', '-sp', default="low_velocity",
                                   help="how to split up gestures <low_velocity, sentence, wordcount, timeconst>")
    parser.add_argument('--wordcount', '-wc', default=10,
                                   help="if split gestures by wordcount, how many words")
    parser.add_argument('--timeconst', '-tc', default=10,
                                   help="if split gestures by time, how many seconds")
    parser.add_argument('--create_videos', '-cv', default=True,
                                   help="should we turn bvh into nice mp4s (or just do split/trasncript)")

    params = parser.parse_args()
    bvh_name = os.path.join(params.raw_dir, 'Motion', params.file_name + '.bvh')
    wav_name = os.path.join(params.raw_dir, 'Audio', params.file_name + '.wav')
    txt_name = os.path.join(params.raw_dir, 'Transcripts', params.file_name + '.json')

    dest_dir = params.dest_dir
    if not os.path.exists(dest_dir):
        os.mkdir(dest_dir)

    # split up the files into new bvh, wav, and json files
    print("splitting bvh files")
    split_bvh_files(bvh_name, txt_name, wav_name, dest_dir, output_name=params.file_name, split_by=params.split_by, wordcount=params.wordcount, timeconst=params.timeconst)

    # collect the transcripts from those files
    print("collecting transcripts")
    collect_transcript_from_dir(dest_dir, 'transcripts.txt')

    print("getting BERT embeddings")
    get_embeddings_for_dir(dest_dir)

    # get histograms
    # can probably do this a little more cleverly.
    print("analyzing files")
    fns = os.listdir(dest_dir)
    fns = [os.path.join(dest_dir, f) for f in fns]
    transcript_files = [f for f in fns if f.endswith('.json')]
    transcript_files.sort(key=sorter)
    get_wc_histogram(transcript_files, dest_dir)

    audio_files = [f for f in fns if f.endswith('.wav')]
    get_time_histogram(transcript_files, dest_dir)

    if params.create_videos != "False" or params.create_videos != "F":
        # now can resample the bvh
        print("resampling bvh files")
        mirror_downsample_bvh(dest_dir, fps=20)

        # now get the mp4 videos visualized from
        print("getting mp4s -- this takes a long time")
        fs = [f for f in os.listdir(dest_dir) if f.endswith('bvh')]  # just test this for now!!
        for f in tqdm(fs):
            full_path = os.path.join(dest_dir, f)
            output = Path(full_path).with_suffix(".mp4")
            print(output)
            bvh_to_video(Path(full_path), output, server_url=LOCAL_SERVER_URL)

        print("marrying matching mp4 and wav")
        combine_av_for_dir(dest_dir)
