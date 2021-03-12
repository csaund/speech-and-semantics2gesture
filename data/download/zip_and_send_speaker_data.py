## necessary because I cannot install youtube_dl on remote machines, so
## must instead download youtube videos onto local machine, zip them up, scp them,
## then delete them locally.


## This script takes a LOOOONG time to run, but does the following:
## - sees if speaker videos are downloaded. If they are not downloaded in Gestures/<speaker>/videos/, downloads them

# because it must also transcribe the videos, it does the following:
# - for all the videos, checks if there is already a transcription in audio_transcript_bucket_1
#   - if that transcription exists, it's done.
#   - if it does not exist, carries on with...
# - scrapes audio from the video using the upload_audio_from_videos script
# - transcribes audio in cloud using the transcribe_audio_in_cloud script

## then finally, it
## zips up videos, sends to carolyns@ccn00.psy.gla.ac.uk:/analyse/Project0325/data/

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
import csv
from download_youtube import download_vids
from upload_audio_from_videos import create_audio_path, scrape_audio_from_videos, upload_audio_to_gcloud
from transcribe_audio_in_cloud import google_transcribe, words_to_csv

parser = argparse.ArgumentParser()
parser.add_argument('-base_path', '--base_path', help='base folder path of dataset')
parser.add_argument('-speaker', '--speaker',
                    help='download videos of a specific speaker {oliver, jon, conan, rock, chemistry, ellen, almaram, angelica, seth, shelly}')
args = parser.parse_args()

videos_path = os.path.join(args.base_path, args.speaker, 'videos')

BASE_PATH = args.base_path
df = pd.read_csv(os.path.join(BASE_PATH, "videos_links.csv"))


TRANSCRIPT_BUCKET = "audio_transcript_buckets_1"
AUDIO_OUTPUT_PATH = "tmp_audio"
TMP_CSV_PATH = "tmp_csv_files"


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


def upload_blob(bucket_name, source_file_name, destination_blob_name):
    """Uploads a file to the bucket."""
    storage_client = storage.Client()
    bucket = storage_client.get_bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)
    blob.upload_from_filename(source_file_name)


# if videos are already downloaded
if not os.path.exists(videos_path) or len(os.listdir(videos_path)) > 0:
    # download the youtube videos!
    download_vids(df)

# videos should all be downloaded to videos_path, now just check if there is a transcript
csv_fns = [v.replace('mkv', 'csv').replace('mp4', 'csv') for v in os.listdir(videos_path)]
existing_transcripts = list_blobs(TRANSCRIPT_BUCKET)
must_transcribe = False

for fn in csv_fns:
    if fn not in existing_transcripts:
        # for now, we just assume one missing one is the same as the whole speaker hasn't been transcribed.
        must_transcribe = True

if must_transcribe:
    # scrape and upload the audio to gcloud
    create_audio_path(args.output_path)
    scrape_audio_from_videos()
    audio_list = os.listdir(AUDIO_OUTPUT_PATH)
    upload_audio_to_gcloud(audio_list, AUDIO_OUTPUT_PATH, bucket_name='audio_bucket_rock_1')

    # transcribe once it's in the cloud
    print("transcribing audio in cloud")
    for audio_fn in tqdm(audio_list):
        # transcribe the file we previously scraped and uploaded
        words = google_transcribe(audio_fn, os.path.join(AUDIO_OUTPUT_PATH, audio_fn))

        # save the words to a csv we can then upload.
        csv_name = audio_fn.replace('wav', 'csv')
        csv_path = os.path.join(args.csv_output_path, csv_name)
        words_to_csv(words, csv_path=csv_path)

        # Upload transcript to cloud
        upload_blob(TRANSCRIPT_BUCKET, csv_path, csv_name)


