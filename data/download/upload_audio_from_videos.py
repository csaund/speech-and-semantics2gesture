from __future__ import unicode_literals

import os
import argparse
import subprocess
import json
## sound stuff
import wave
from pydub import AudioSegment
## Google stuff
from google.cloud import storage
import shutil

devKey = str(open(os.path.join(os.getenv("HOME"), "devKey"), "r").read()).strip()
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = os.path.join(os.getenv("HOME"), 'google-creds.json')

parser = argparse.ArgumentParser(description='Description of your program')
parser.add_argument('-b', '--base_dataset_path', help="dataset root path")
parser.add_argument('-np', '--num_processes', type=int, default=1)
parser.add_argument('-s', '--speaker', default=None)
parser.add_argument('-o', '--output_path', default=None)
parser.add_argument('-sc', '--scrape', action='store_false')
parser.add_argument('-u', '--upload', action='store_false')
args = parser.parse_args()

VIDEOS_PATH = os.path.join(args.base_dataset_path, args.speaker, 'videos')
AUDIO_OUTPUT_PATH = args.output_path
AUDIO_FN_TEMPLATE = os.path.join(args.base_dataset_path, '%s', 'train', 'audio', '%s_%s_%s-%s.wav')
GCS_AUDIO_TEMPLATE = 'gs://' + args.speaker + '/'
TEMP_AUDIO_PATH = 'temp_audio_to_delete'

from apiclient.discovery import build
service = build('language', 'v1', developerKey=devKey)
collection = service.documents()


# From speaker with videos downloaded (basepath/Gestures/<speaker>/videos), scrapes audio from videos and
# uploads to google cloud where audio lives. Audio will live in
# audio_bucket_rock_1 because I am incredibly lazy

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


# Goes into <base_path>/<speaker>/videos and scrapes audio
# stores in google cloud
def scrape_audio_from_videos():
    videos_list = os.listdir(VIDEOS_PATH)
    for video_fn in videos_list:
        video_path = os.path.join(VIDEOS_PATH, video_fn)
        output_audio_path = os.path.join(AUDIO_OUTPUT_PATH, video_fn)
        command = ("ffmpeg -i %s -ab 160k -ac 2 -ar 48000 -vn %s" % (video_path, output_audio_path))
        proc = subprocess.Popen(command, shell=True)
        proc.wait()


def upload_audio_to_gcloud():
    audios_list = os.listdir(AUDIO_OUTPUT_PATH)
    # create_audio_path(TEMP_AUDIO_PATH)
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
    # remove_audio_path(TEMP_AUDIO_PATH)


def create_audio_path(op):
    if not os.path.exists(op):
        os.mkdir(op)


def remove_audio_path(op):
    shutil.rmtree(op)


print("args: ", args.scrape, args.upload)
if args.scrape is not None:
    print("")
    print("trying to scrape")
    print("")
    create_audio_path(args.output_path)
    scrape_audio_from_videos()
if args.upload is not None:
    print("")
    print("trying to upload")
    print("")
    upload_audio_to_gcloud()