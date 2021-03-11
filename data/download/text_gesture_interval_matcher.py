from __future__ import unicode_literals

import argparse
import subprocess
import re
from subprocess import call

import cv2
import numpy as np
import os
import shutil
import pandas as pd
from tqdm import tqdm
import wave
from pydub import AudioSegment
from google.cloud import storage
from google.cloud import speech
import datalab.storage as gcs
# TODO : CARO GET RID OF THIS
from google.datalab import Context as ctx
import csv
from tqdm import tqdm, tqdm_pandas
tqdm_pandas(tqdm())

## the module that takes in intervals_df and looks in the transcript to assign
## the raw text that matches that gesture, and does further semantic mapping of that text
## and outputs the resulting csv into intervals_with_semantics_df.csv.


## reads intervals_df into pd df
## Downloads transcript from google cloud
## Does some logic over whether that transcript is old transcription or a new one.
## matches timestamps of transcript to timestamps of gestures
## Does semantic analysis of the transcript segment that matches that gesture.
## outputs result into intervals_with_semantics_df.csv
## which contains fp to audio, video, and npz (keypoint) data.
# Thus outputs entire


ctx.project_id = 'scenic-arc-250709'

parser = argparse.ArgumentParser()
parser.add_argument('-train_csv', '--train_csv', help='where the training csv lives')
parser.add_argument('-base_path', '--base_path', help='where is the base path for TRAINING data? is it Gestures/<speaker> ?')
parser.add_argument('-speaker', '--speaker',
                    help='download videos of a specific speaker {oliver, jon, conan, rock, chemistry, ellen, almaram, angelica, seth, shelly}')
args = parser.parse_args()

TRANSCRIPT_BUCKET = "audio_transcript_buckets_1"
TMP_CSV_PATH = "tmp_csv_files"
BASE_PATH = args.base_path

## input looks like this:
# dataset,start,end,interval_id,pose_fn,audio_fn,video_fn,speaker
# dataset: train,
# start: 00:09:10.583917,
# end: 00:09:14.788121,
# interval_id: 120145,
# pose_fn: Gestures\almaram\train\npz\120145-00_09_10.583917-00_09_14.788121.npz,
# audio_fn: Gestures\almaram\train\audio\almaram_Lessons_in_Fiqh_17-Xp-2bPf2woc.mkv_00_09_10.583917-00_09_20.460460.wav,
# video_fn: Lessons_in_Fiqh_17-Xp-2bPf2woc.mkv,
# speaker: almaram


def convert_timestamp_to_seconds(timestamp):
    hours, minutes, seconds = timestamp.split(":")
    time = (float(hours) * 3600) + (float(minutes) * 60) + float(seconds)
    return time


def get_csv_name_from_vfn(vfn, ext='csv'):
    return vfn.replace('mkv', ext).replace('mp4', ext)


def extract_words_from_transcript(word_dict):
    words = []
    for w in word_dict:
        words.append(w.word)

def add_transcript_to_row(row):
    if row.transcript is None:
        csv_fn = os.path.join(TMP_CSV_PATH, get_csv_name_from_vfn(row.video_fn))
        if not os.path.exists(csv_fn):
            print("WARNING failed to download transcript", csv_fn)
            return row.transcript
        words_df = pd.read_csv(csv_fn)
        start_time = convert_timestamp_to_seconds(row.start)
        end_time = convert_timestamp_to_seconds(row.end)
        relevant_words = words_df.loc[(words_df['start_time'] >= start_time) & (words_df['end_time'] <= end_time)]
        relevant_words = relevant_words.sort_values(by='start_time')  # just in case the words aren't in order for some reason
        transcript = ' '.join(list(relevant_words.word))     # Create the transcript string
        return transcript
    else:
        print("think row transcript was not none: ", row.transcript)
    return row.transcript


def add_semantic_analysis_to_row(row):
    # do wn stuff here
    # add the analysis
    # win the day
    raise NotImplementedError


def download_blob(bucket_name, source_blob_name, destination_file_name):
    """Downloads a blob from the bucket."""
    storage_client = storage.Client()
    bucket = storage_client.get_bucket(bucket_name)
    blob = bucket.blob(source_blob_name)
    blob.download_to_filename(destination_file_name)
    print('Blob {} downloaded to {}.'.format(
        source_blob_name,
        destination_file_name))


def list_blobs(bucket_name):
    """Lists all the blobs in the bucket."""
    # bucket_name = "your-bucket-name"
    storage_client = storage.Client()
    # Note: Client.list_blobs requires at least package version 1.17.0.
    blobs = storage_client.list_blobs(bucket_name)
    f_names = []
    for blob in blobs:
        f_names.append(blob.name)
    return f_names


# Holy heck it works.
# TODO: make it output to the training spot instead of cur working dir
if __name__ == "__main__":
    df = pd.read_csv(args.train_csv)
    if 'transcript' not in df.columns:
        df["transcript"] = None
        df["semantic_analysis"] = None

    # video_fn in train.csv can be swapped out to get the transcript file from google cloud.
    # get all the video_fns and download all the transcript csvs.
    video_fns = df.video_fn.unique()
    # get csv names to download
    csv_fns = [get_csv_name_from_vfn(v) for v in video_fns]

    if not os.path.exists(TMP_CSV_PATH):
        os.mkdir(TMP_CSV_PATH)

    possible_transcripts = list_blobs(TRANSCRIPT_BUCKET)
    # for all the transcripts in the transcript bucket
    for fn in csv_fns:
        if fn not in possible_transcripts:
            print("WARNING no transcript found for ", fn)
        # if we haven't already downloaded it
        if not os.path.exists(os.path.join(TMP_CSV_PATH, fn)):
            download_blob(TRANSCRIPT_BUCKET, fn, os.path.join(TMP_CSV_PATH, fn))

    tqdm.pandas()       # watch this bad boy
    df['transcript'] = df.progress_apply(lambda row: add_transcript_to_row(row), axis=1)
    # df['semantic_analysis'] = df.progress_apply(lambda row: add_semantic_analysis_to_row(row), axis=1)

    output_csv = 'training_with_semantics.csv'
    df.to_csv(os.path.join(BASE_PATH, output_csv))

    shutil.rmtree(TMP_CSV_PATH)