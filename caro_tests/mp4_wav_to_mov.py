from argparse import ArgumentParser
import moviepy.editor as mpe

from caro_tests.bvh_helpers import get_positions, get_low_velocity_hand_points, timestr_to_float, get_times_of_splits
from pydub import AudioSegment
import json

def combine_audio(vidname, audname, outname, fps=25):
    my_clip = mpe.VideoFileClip(vidname)
    audio_background = mpe.AudioFileClip(audname)
    final_clip = my_clip.set_audio(audio_background)
    final_clip.write_videofile(outname, fps=fps)


if __name__ == "__main__":
    # Setup parameter parser
    parser = ArgumentParser()
    parser.add_argument('--wav_file', '-wav', default="",
                                   help="wav file to apply to video")
    parser.add_argument('--vid_file', '-vid', default="",
                                   help="mp4 file to apply to video")
    parser.add_argument('--output', '-out', default="",
                                   help="output name of mp4")

    params = parser.parse_args()
    combine_audio(params.vid_file, params.wav_file, params.output)