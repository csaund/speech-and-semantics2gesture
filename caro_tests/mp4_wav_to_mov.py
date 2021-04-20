from argparse import ArgumentParser
import moviepy.editor as mpe
import os
from pathlib import Path


def get_matching_wav_mp4s(dir):
    files = os.listdir(dir)


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
    parser.add_argument('--dir', '-d', default=None,
                                   help="directory containing matching mp4 and wav names to marry")

    params = parser.parse_args()

    if params.dir:
        # get matching wav/mp4 files
        wavs, mp4s = get_matching_wav_mp4s(params.dir)
        for i in range(len(wavs)):
            outname = Path(wavs[i]).with_suffix('_sound.mp4')
            combine_audio(mp4s[i], wavs[i], outname)
    combine_audio(params.vid_file, params.wav_file, params.output)