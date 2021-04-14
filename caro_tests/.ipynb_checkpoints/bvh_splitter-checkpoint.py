import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from argparse import ArgumentParser

import glob
import os
import sys
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)

from pymo.parsers import BVHParser
from pymo.data import Joint, MocapData
from pymo.preprocessing import *
from pymo.writers import *
from pymo.features import *
from pymo.preprocessing import *
from pymo.viz_tools import *

import joblib as jl

RIGHT_HAND_X = 'RightHand_Xposition'
LEFT_HAND_X = 'LeftHand_Xposition'
RIGHT_HAND_Y = 'RightHand_Yposition'
LEFT_HAND_Y = 'LeftHand_Yposition'
RIGHT_HAND_Z = 'RightHand_Zposition'
LEFT_HAND_Z = 'LeftHand_Zposition'


LEFT_HAND = {
    'x': LEFT_HAND_X,
    'y': LEFT_HAND_Y,
    'z': LEFT_HAND_Z
}

RIGHT_HAND = {
    'x': RIGHT_HAND_X,
    'y': RIGHT_HAND_Y,
    'z': RIGHT_HAND_Z
}

## the goal of this file is to identify where to cut the gestures
## the criteria for that include:
## when both hands have a velocity below a certain threshold
## when both hands are in the rest position?


# takes list of frames to split bvh file at
# saves as many files as splits+1
def split_bvh_at_frames(orig_bvh, frame_splits):
    f = open(orig_bvh, "r")

    # create files for split
    splits = []
    prev_frame = 0
    for i in range(0, len(frame_splits)+1):
        if i == len(frame_splits):
            fn = f'orig_bvh_split_{i}_frame_{prev_frame}_+.bvh'
            splits.append(open(fn, 'wt'))
        else:
            fn = f'orig_bvh_split_{i}_frame_{prev_frame}_{frame_splits[i]}.bvh'
            splits.append(open(fn, 'wt'))
            prev_frame = frame_splits[i]
        print(fn)
    # 15
    # copy hierarchy
    line = ""
    while 'MOTION' not in str(line):
        line = f.readline()
        for s in splits:
            s.write(str(line))

    # get the frame counts for each split
    framecounts = []
    line = str(f.readline())
    orig_framecount = int(line.split("\t")[1])
    for i in range(0, len(frame_splits)+1):
        if i == 0:
            framecounts.append(frame_splits[i])
        elif i < len(frame_splits):
            framecounts.append(frame_splits[i] - frame_splits[i-1])
        else:
            framecounts.append(orig_framecount - frame_splits[-1])
    for i in range(len(splits)):
        splits[i].write(str('Frames: %s\n' % framecounts[i]))
    # 36
    # add the frame time
    line = f.readline()
    for s in splits:
        s.write(str(line))
    # 41
    # add frames as we go
    cur_framecount = 0
    i = 0
    print(frame_splits)
    while i < len(splits):
        line = ""
        print("i!", i)
        if i == len(frame_splits):
            print('got to the end: ')
            while f:
                line = f.readline()
                splits[i].write(str(line))
                if line == "":
                    break
        else:
            print('cur framecount: ', cur_framecount)
            print('frame split: ', frame_splits[i])
            while cur_framecount < frame_splits[i]:
                line = f.readline()
                splits[i].write(str(line))
                cur_framecount += 1
        # print(str(line))
        i += 1

    f.close()
    for s in splits:
        s.close()


def extract_joint_angles(bvh_file, fps):
    print('extract joint angles')
    p = BVHParser()

    data_all = []
    ff = bvh_file
    print(ff)
    data_all.append(p.parse(ff))
    print('got all data')

    data_pipe = Pipeline([
       ('dwnsampl', DownSampler(tgt_fps=fps,  keep_all=False)),
       ('root', RootTransformer('hip_centric')),
       ('mir', Mirror(axis='X', append=True)),
       ('jtsel', JointSelector(['Spine','Spine1','Spine2','Spine3','Neck','Neck1','Head','RightShoulder', 'RightArm', 'RightForeArm', 'RightHand', 'LeftShoulder', 'LeftArm', 'LeftForeArm', 'LeftHand'], include_root=True)),
       ('exp', MocapParameterizer('expmap')),
       ('cnst', ConstantsRemover()),
       ('np', Numpyfier())
    ])

    out_data = data_pipe.fit_transform(data_all)
    return out_data


def get_positions(bvh_file):
    p = BVHParser()
    parsed_data = p.parse(bvh_file)
    mp = MocapParameterizer('position')
    positions = mp.fit_transform([parsed_data])
    return positions


def get_contact_idxs(signal, t=0.02, min_dist=120):
    up_idxs = peakutils.indexes(signal, thres=t/max(signal), min_dist=min_dist)
    down_idxs = peakutils.indexes(-signal, thres=t/min(signal), min_dist=min_dist)
    return [up_idxs, down_idxs]


def plot_up_down(mocap_track, col_name=RIGHT_HAND_Y, t=0.02, min_dist=120, signal=None):
    if signal is None:
        signal = mocap_track.values[col_name].values
    idxs = get_contact_idxs(signal, t, min_dist)

    plt.plot(mocap_track.values.index, signal)
    plt.plot(mocap_track.values.index[idxs[0]], signal[idxs[0]], 'ro')
    plt.plot(mocap_track.values.index[idxs[1]], signal[idxs[1]], 'go')


def plot_wrist_positions(modat):
    plot_up_down(modat, RIGHT_HAND_Y)
    plot_up_down(modat, LEFT_HAND_Y)


def plot_wrist_velocities(modat):
    l_signal = get_hand_velocity(modat, LEFT_HAND)
    r_signal = get_hand_velocity(modat, RIGHT_HAND)
    plt.plot(modat.values.index, np.array(l_signal))
    plt.plot(modat.values.index, np.array(r_signal))


def get_hand_velocity(modat, hand=LEFT_HAND):
    xs, ys, zs = modat.values[hand['x']].values, modat.values[hand['y']].values, modat.values[hand['z']].values,
    vels = [get_dist(np.array([xs[i], ys[i], zs[i]]), np.array([xs[i+1], ys[i+1], zs[i+1]])) for i in range(0, len(xs)-1)]
    vels.append(0)
    return vels


def get_dist(p1, p2):
    squared = np.sum((p1-p2)**2, axis=0)
    return np.sqrt(squared)


def play_motion_data(modat):
    disp = nb_play_mocap(modat, 'pos',
        scale = 2, camera_z = 800, frame_time = 1 / 120,
        base_url = 'pymo/mocapplayer/playBuffer.html')



if __name__ == "__main__":
    # Setup parameter parser
    parser = ArgumentParser()
    parser.add_argument('--bvh_dir', '-orig', default="../../dataset/raw_data/Motion",
                                   help="Path where original motion files (in BVH format) are stored")
    parser.add_argument('--dest_dir', '-dest', default="../../dataset/raw_data/Motion",
                                   help="Path where extracted motion features will be stored")
    parser.add_argument('--pipeline_dir', '-pipe', default="../utils/",
                        help="Path where the motion data processing pipeline will be stored")
    parser.add_argument('--bvh_file', '-bvh', default="NaturalTalking_010.bvh",
                        help="bvh file to extract")

    params = parser.parse_args()
    print('params: ', params)
    extract_joint_angles(params.bvh_file, fps=60)