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

ctx.project_id = 'scenic-arc-250709'

parser = argparse.ArgumentParser()
parser.add_argument('-base_path', '--base_path', help='base folder path of dataset')
parser.add_argument('-audio_path', '--audio_path', help='path to local audio files to transcribe (folder must contain filenames that match the gcloud transcript names')
parser.add_argument('-speaker', '--speaker',
                    help='download videos of a specific speaker {oliver, jon, conan, rock, chemistry, ellen, almaram, angelica, seth, shelly}')
args = parser.parse_args()

video_iterator = 0      # disgusting hack

AUDIO_INPUT_PATH = args.audio_path
AUDIO_BUCKET = "audio_bucket_rock_1"
TRANSCRIPT_BUCKET = "transcript_buckets_1"

storage_client = storage.Client()
devKey = str(open("%s/devKey" % os.getenv("HOME"), "r").read()).strip()

from apiclient.discovery import build
service = build('language', 'v1', developerKey=devKey)
collection = service.documents()

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "%s/google-creds.json" % os.getenv("HOME")
print("literally anything")

# Then, transcribes the audio into a separate bucket.
def stereo_to_mono(audio_file_path):
    sound = AudioSegment.from_wav(audio_file_path)
    sound = sound.set_channels(1)
    sound.export(audio_file_path, format="wav")


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


def google_transcribe(fn, fp):
    print("attempting to transcribe file %s" % fn)

    # check if transcript is already there
    transcript_bucket = storage_client.bucket(TRANSCRIPT_BUCKET)
    transcript_file_exists_in_cloud = storage.Blob(bucket=transcript_bucket, name=fn).exists(storage_client)
    if transcript_file_exists_in_cloud:
        print("Already transcribed audio from ", fn)
        return

    gcs_uri = 'gs://' + AUDIO_BUCKET + '/' + fn

    # frame_rate, channels = frame_rate_channel(fp)
    client = speech.SpeechClient()
    audio = speech.RecognitionAudio(uri=gcs_uri)
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.FLAC,
        sample_rate_hertz=16000,
        language_code='en-US',
        enable_word_time_offsets=True,
        enable_automatic_punctuation=True)

    # Detects speech in the audio file
    operation = client.long_running_recognize(config=config, audio=audio)
    print("getting results...")
    response = operation.result(timeout=1000)

    # Each result is for a consecutive portion of the audio. Iterate through
    # them to get the transcripts for the entire audio file.
    transcript = {}
    i = 0
    for result in response.results:
        i += 1
        print("result: ", result)
        print(u"Transcript: {}".format(result.alternatives[0].transcript))
        print("Confidence: {}".format(result.alternatives[0].confidence))
        out = {
            'transcript': result.alternatives[0].transcript,
            'confidence': result.alternatives[0].confidence
        }
        transcript[i] = out

    ## TODO uncomment/implement if I want to do this.
    # this resource is brilliant:
    # https://towardsdatascience.com/how-to-use-google-speech-to-text-api-to-transcribe-long-audio-files-1c886f4eb3e9
    # delete_blob(bucket_name, destination_blob_name)
    print("got transcript", transcript)
    return pd.DataFrame.from_dict(transcript)


def upload_transcript_to_gcloud():
    audios_list = os.listdir(AUDIO_OUTPUT_PATH)

    for audio_fn in audios_list:
        destination_bucket = 'audio_bucket_rock_1'
        destination_name = audio_fn
        print("uploading %s to %s" % (audio_fn, destination_bucket))
        audio_fp = os.path.join(AUDIO_OUTPUT_PATH, audio_fn)
        # frame_rate, channels = frame_rate_channel(audio_fp)
        # if channels > 1:
        #     stereo_to_mono(audio_fp)

        # upload so we can get a gcs for long audio transcription.
        upload_blob(destination_bucket, audio_fp, destination_name)

if __name__ == "__main__":
    audios_list = os.listdir(AUDIO_INPUT_PATH)
    # make df that we can then pickle and upload to cloud
    for audio_fn in audios_list:
        print("this should fully print first.")
        transcript_df = google_transcribe(audio_fn, os.path.join(AUDIO_INPUT_PATH, audio_fn))
        # Upload transcript to cloud
        gcs.Bucket(TRANSCRIPT_BUCKET).item('to/data.csv').write_to(transcript_df.to_csv(), 'text/csv')
