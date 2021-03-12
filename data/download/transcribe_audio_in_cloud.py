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

ctx.project_id = 'scenic-arc-250709'

parser = argparse.ArgumentParser()
parser.add_argument('-csv_output_path', '--csv_output_path', help='where to output the csv files')
parser.add_argument('-audio_path', '--audio_path', help='path to local audio files to transcribe (folder must contain filenames that match the gcloud transcript names')
parser.add_argument('-speaker', '--speaker',
                    help='download videos of a specific speaker {oliver, jon, conan, rock, chemistry, ellen, almaram, angelica, seth, shelly}')
args = parser.parse_args()

AUDIO_INPUT_PATH = args.audio_path
AUDIO_BUCKET = "audio_bucket_rock_1"
TRANSCRIPT_BUCKET = "audio_transcript_buckets_1"

storage_client = storage.Client()
devKey = str(open("%s/devKey" % os.getenv("HOME"), "r").read()).strip()

from apiclient.discovery import build
service = build('language', 'v1', developerKey=devKey)
collection = service.documents()

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "%s/google-creds.json" % os.getenv("HOME")


def frame_rate_channel(audio_file_path):
    wav_file = wave.open(audio_file_path, "rb")
    frame_rate = wav_file.getframerate()
    channels = wav_file.getnchannels()
    return frame_rate, channels


def upload_blob(bucket_name, source_file_name, destination_blob_name):
    """Uploads a file to the bucket."""
    storage_client = storage.Client()
    bucket = storage_client.get_bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)
    blob.upload_from_filename(source_file_name)


# converts of list of dicts with the same keys to a csv file.
def words_to_csv(toCSV, csv_path):
    keys = toCSV[0].keys()
    with open(csv_path, 'w', newline='') as output_file:
        dict_writer = csv.DictWriter(output_file, keys)
        dict_writer.writeheader()
        dict_writer.writerows(toCSV)


def google_transcribe(fn, fp):
    print("attempting to transcribe file %s" % fn)

    # check if transcript is already there
    transcript_bucket = storage_client.bucket(TRANSCRIPT_BUCKET)
    transcript_file_exists_in_cloud = storage.Blob(bucket=transcript_bucket, name=fn).exists(storage_client)
    if transcript_file_exists_in_cloud:
        print("Already transcribed audio from ", fn)
        return

    gcs_uri = 'gs://' + AUDIO_BUCKET + '/' + fn

    frame_rate, channels = frame_rate_channel(fp)
    client = speech.SpeechClient()
    audio = speech.RecognitionAudio(uri=gcs_uri)
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=frame_rate,
        language_code='en-US',
        enable_word_time_offsets=True,
        enable_automatic_punctuation=True)

    # Detects speech in the audio file
    operation = client.long_running_recognize(config=config, audio=audio)
    response = operation.result(timeout=1000)

    # Each result is for a consecutive portion of the audio. Iterate through
    # them to get the transcripts for the entire audio file.
    words = []
    for result in response.results:
        for w in result.alternatives[0].words:
            words.append({
                'word': w.word,
                'start_time': w.start_time.seconds + (w.start_time.microseconds / 1000000),  # dis some bullshit where it looks like it's nanos
                'end_time': w.end_time.seconds + (w.end_time.microseconds / 1000000)        # but its a datetime obj so it's microseconds.
            })

    # this resource is brilliant:
    # https://towardsdatascience.com/how-to-use-google-speech-to-text-api-to-transcribe-long-audio-files-1c886f4eb3e9
    return words


# works but overall a bit hacky re: bucket names, gc permissions, etc
if __name__ == "__main__":
    audios_list = os.listdir(AUDIO_INPUT_PATH)
    for audio_fn in tqdm(audios_list):
        # transcribe the file we previously scraped and uploaded
        words = google_transcribe(audio_fn, os.path.join(AUDIO_INPUT_PATH, audio_fn))

        # save the words to a csv we can then upload.
        csv_name = audio_fn.replace('wav', 'csv')
        csv_path = os.path.join(args.csv_output_path, csv_name)
        words_to_csv(words, csv_path=csv_path)

        # Upload transcript to cloud
        upload_blob(TRANSCRIPT_BUCKET, csv_path, csv_name)
