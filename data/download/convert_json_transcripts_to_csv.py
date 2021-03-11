# I biffed before when I was getting all the transcripts
# and now they're saved in a super shitty json format.
# in this script I fix it and go from a shittily formatted json
# to a nice word csv transcript for each video.
# this shit is supposed to be run once and never again.

import argparse
import os
import pandas as pd
from google.cloud import storage
from google.datalab import Context as ctx
import csv
from tqdm import tqdm, tqdm_pandas
import shutil


ctx.project_id = 'scenic-arc-250709'

TRANSCRIPT_BUCKET = "audio_transcript_buckets_1"
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


# hmm, slightly different...
def words_to_csv(toCSV, csv_path):
    keys = toCSV.keys()
    with open(csv_path, 'w', newline='') as output_file:
        dict_writer = csv.DictWriter(output_file, keys)
        dict_writer.writeheader()
        dict_writer.writerows(toCSV)


if __name__ == "__main__":
    possible_transcripts = list_blobs(TRANSCRIPT_BUCKET)
    if not os.path.exists(TMP_CSV_PATH):
        os.mkdir(TMP_CSV_PATH)
    for trans in tqdm(possible_transcripts):
        # convert this garbage.
        if '.json' in trans:
            csv_name = trans.replace('json', 'csv')
            download_blob(TRANSCRIPT_BUCKET, trans, os.path.join(TMP_CSV_PATH, trans))
            df = pd.read_json(os.path.join(TMP_CSV_PATH, trans))

            # structure of this is transcript, words where words has all the information... we basically want to
            # flatten this out and convert it.

            # ugh actually this is so slow. Just going to convert it the super slow way
            # cause we only need to do it once.
            words_df = pd.DataFrame(columns=['word', 'start_time', 'end_time'])
            for ind, row in df.iterrows():
                word = []
                start_time = []
                end_time = []
                for w in row.words:
                    word.append(w['word'])
                    start_time.append(w['word_start'])
                    end_time.append(w['word_end'])
                ap_df = pd.DataFrame(list(zip(word, start_time, end_time)), columns=['word', 'start_time', 'end_time'])
                words_df = words_df.append(ap_df)
            csv_path = os.path.join(TMP_CSV_PATH, csv_name)
            words_df.to_csv(csv_path)
            upload_blob(TRANSCRIPT_BUCKET, csv_path, csv_name)

    # TODO delete temp csv path
    shutil.rmtree(TMP_CSV_PATH)
