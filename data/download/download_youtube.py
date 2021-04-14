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

parser = argparse.ArgumentParser()
parser.add_argument('-base_path', '--base_path', help='base folder path of dataset')
parser.add_argument('-speaker', '--speaker',
                    help='download videos of a specific speaker {oliver, jon, conan, rock, chemistry, ellen, almaram, angelica, seth, shelly}')
args = parser.parse_args()

BASE_PATH = args.base_path
df = pd.read_csv(os.path.join(BASE_PATH, "videos_links.csv"))

if args.speaker:
    df = df[df['speaker'] == args.speaker]

temp_output_path = os.path.join('tmp', 'temp_video.mp4')


handle_pat = re.compile(r'(.*?)\s+pid:\s+(\d+).*[0-9a-fA-F]+:\s+(.*)')


def open_files(name):
    """return a list of (process_name, pid, filename) tuples for
       open files matching the given name."""
    lines = subprocess.check_output('handle.exe "%s"' % name).splitlines()
    results = (_handle_pat.match(line.decode('mbcs')) for line in lines)
    return [m.groups() for m in results if m]


def download_vids(df):
    video_iterator = 0  # disgusting hack
    for _, row in tqdm(df.iterrows(), total=df.shape[0]):
        i, name, link = row
        video_iterator += 1
        temp_output_path_iterated = temp_output_path + '_' + str(video_iterator)           # so gross

        if 'youtube' in link:
            proc1, proc2 = None, None           # shite practice but c'mon we gotta do it.
            try:
                output_path = os.path.join(BASE_PATH, row["speaker"], "videos", row["video_fn"])
                if not (os.path.exists(os.path.dirname(output_path))):
                    os.makedirs(os.path.dirname(output_path))
                command = 'youtube-dl -o {temp_path} -f mp4 {link}'.format(link=link, temp_path=temp_output_path_iterated)
                proc1 = subprocess.Popen(command, shell=True)
                proc1.wait()
                proc1.terminate()
                cam = cv2.VideoCapture(temp_output_path_iterated)
                if np.isclose(cam.get(cv2.CAP_PROP_FPS), 29.97, atol=0.03):
                    shutil.move(temp_output_path_iterated, output_path)
                else:
                    proc2 = subprocess.Popen('ffmpeg -i "%s" -r 30000/1001 -strict -2 "%s" -y' % (temp_output_path_iterated, output_path),
                                shell=True)
                    proc2.wait()
                    proc2.terminate()
            except Exception as e:
                print("Exception: ", e)
            finally:
                if os.path.exists(temp_output_path):
                    poll = proc1.poll() or proc2.poll()
                    if proc1:
                        proc1.terminate()
                    if proc2:
                        proc2.terminate()
                    f = open(temp_output_path, 'r')
                    f.close()
                    print(open_files(temp_output_path))
                    os.remove(temp_output_path)
                else:
                    print("don't think this path exists: ", temp_output_path)
    print("Out of a total of %s videos for %s: "%(len(df), args.speaker))
    print("Successfully downloaded:")
    my_cmd = 'ls ' + os.path.join(BASE_PATH, row["speaker"], "videos") + ' | wc -l'
    os.system(my_cmd)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-base_path', '--base_path', help='base folder path of dataset')
    parser.add_argument('-speaker', '--speaker',
                        help='download videos of a specific speaker {oliver, jon, conan, rock, chemistry, ellen, almaram, angelica, seth, shelly}')
    args = parser.parse_args()

    BASE_PATH = args.base_path
    df = pd.read_csv(os.path.join(BASE_PATH, "videos_links.csv"))

    if args.speaker:
        df = df[df['speaker'] == args.speaker]

    temp_output_path = os.path.join('tmp', 'temp_video.mp4')
    download_vids(df)
