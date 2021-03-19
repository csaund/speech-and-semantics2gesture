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
from common.google_helpers import list_blobs, upload_blob
from tqdm import tqdm
import pandas as pd

devKey = str(open(os.path.join(os.getenv("HOME"), "devKey"), "r").read()).strip()
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = os.path.join(os.getenv("HOME"), 'google-creds.json')


from apiclient.discovery import build
service = build('language', 'v1', developerKey=devKey)
collection = service.documents()


parser = argparse.ArgumentParser(description='Description of your program')
parser.add_argument('-b', '--base_path', help="dataset root path")
parser.add_argument('-s', '--speaker', default=None)
args = parser.parse_args()

## from speaker data, checks how many video's audio is uploaded
## and from that, how many of those audio files are transcribed.

AUDIO_BUCKET = 'audio_bucket_rock_1'
TRANSCRIPT_BUCKET = 'audio_transcript_buckets_1'

if __name__ == "__main__":
    df = pd.read_csv(os.path.join(args.base_path, "videos_links.csv"))
    df = df[df['speaker'] == args.speaker]
    speaker_video_links = df['video_fn'].unique()
    print("Got %s videos for speaker %s" % (str(len(speaker_video_links)), args.speaker))
    possible_wavs = [v.replace('mp4', 'wav').replace('webm', 'wav').replace('mkv', 'wav') for v in speaker_video_links]
    possible_transcripts = [w.replace('wav', 'csv') for w in possible_wavs]
    current_wavs = list_blobs(AUDIO_BUCKET)
    current_speaker_wavs = [w for w in possible_wavs if w in current_wavs]
    current_transcripts = list_blobs(TRANSCRIPT_BUCKET)
    current_speaker_transcripts = [t for t in possible_transcripts if t in current_transcripts]
    print('Found %s/%s wav files' % (str(len(current_speaker_wavs)), str(len(possible_wavs))))
    print('Found %s/%s csv files' % (str(len(current_speaker_transcripts)), str(len(possible_transcripts))))
