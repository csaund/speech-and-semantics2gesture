from argparse import ArgumentParser
import moviepy.editor as mpe
import os
from pathlib import Path


# my god this is hacky
def get_file_for_split_number(files, split_num, ext):
    return [f for f in files if 'split_%s' % split_num in f and ext in f][0]


# god disgusting
def get_num_splits(files):
    splits = []
    for f in files:
        try:
            splits.append(int(f.split('_split')[-1].split('_')[1]))
        except Exception as e:
            print('could not get split for file ', f)
    return max(splits)


def get_matching_wav_mp4s(target_dir):
    files = os.listdir(target_dir)
    # need to get names that match up -- need to do this by matching the split number
    wavs = []
    mp4s = []
    num_splits = get_num_splits(files)
    for split_num in range(num_splits + 1):
        try:
            mp4 = [f for f in files if 'split_%s' % split_num in f and '.mp4' in f][0]
            wav = [f for f in files if 'split_%s' % split_num in f and '.wav' in f][0]
            mp4s.append(mp4)
            wavs.append(wav)
        except Exception as e:
            print('could not get mp4 pair: split %s; %s', (split_num, e))

    assert(len(wavs) == len(mp4s))
    return wavs, mp4s


def combine_av(vidname, audname, outname, fps=20):
    my_clip = mpe.VideoFileClip(vidname)
    audio_background = mpe.AudioFileClip(audname)
    final_clip = my_clip.set_audio(audio_background)
    final_clip.write_videofile(outname, fps=fps)


def combine_av_for_dir(target_dir):
    # get matching wav/mp4 files
    wavs, mp4s = get_matching_wav_mp4s(target_dir)
    for i in range(len(mp4s)):
        outname = mp4s[i].split('.mp4')[-2] + '_sound.mp4'
        # create the file?
        if not os.path.exists(outname):
            with open(outname, 'w'): pass
        combine_av(os.path.join(target_dir, mp4s[i]), os.path.join(target_dir, wavs[i]),
                   os.path.join(target_dir, outname))


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
        combine_av_for_dir(params.dir)

    else:
        combine_av(params.vid_file, params.wav_file, params.output)