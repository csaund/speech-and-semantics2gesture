from local_modules.pymo.parsers import BVHParser
from local_modules.pymo.data import Joint, MocapData
from local_modules.pymo.preprocessing import *
from local_modules.pymo.writers import *
from local_modules.pymo.features import *
from local_modules.pymo.preprocessing import *
from local_modules.pymo.viz_tools import *
from sklearn.pipeline import Pipeline
import math

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

BVH_FPS = 0.0166667
VELOCITY_THRESHOLD = 0.15


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


def get_low_velocity_hand_points(modat):
    l_signal = get_hand_velocity(modat, LEFT_HAND)
    r_signal = get_hand_velocity(modat, RIGHT_HAND)
    low_vel_frames = []
    for i in range(len(l_signal)):
        if l_signal[i] < VELOCITY_THRESHOLD and r_signal[i] < VELOCITY_THRESHOLD:
            low_vel_frames.append(i)
    return collapse_low_velocity_points(low_vel_frames)


def collapse_low_velocity_points(low_vels):
    velocity_frames = []
    FRAME_LIMIT = 5

    i = 1
    while i < len(low_vels):
        # while these low velocities are close together
        while low_vels[i] - low_vels[(i-1)] < FRAME_LIMIT:
            i += 1
        # once we're at a point beyond the low velocities
        velocity_frames.append(low_vels[i])
        i += 1
    return velocity_frames


# takes frames at which we split bvh and
# converts to times.
def get_times_of_splits(split_frames):
    return [(f * BVH_FPS) for f in split_frames]


# times comes in s
def get_frames_of_splits(split_times):
    return [math.floor(t / BVH_FPS) for t in split_times]


def timestr_to_float(s):
    return float(s.split('s')[0])


import cv2
import os
import numpy as np

image_folder = 'images'
video_name = 'video.avi'


def save_images(bvh_file, dir='tmp'):
    if not os.path.exists(dir):
        os.mkdir(os.path.join(os.getcwd(), dir))


def mirror_sequence(sequence):
    mirrored_rotations = sequence[:, 1:, :]
    mirrored_trajectory = np.expand_dims(sequence[:, 0, :], axis=1)

    temp = mirrored_rotations.copy()

    # Flip left/right joints
    mirrored_rotations[:, joints_left] = temp[:, joints_right]
    mirrored_rotations[:, joints_right] = temp[:, joints_left]

    mirrored_rotations[:, :, [1, 2]] *= -1
    mirrored_trajectory[:, :, 0] *= -1

    mirrored_sequence = np.concatenate((mirrored_trajectory, mirrored_rotations), axis=1)

    return mirrored_sequence


def sorter(w):
    try:
        n = w.split('_')
        for i in range(len(n)):
            if n[i] == 'split':
                return int(n[i+1])
    except Exception as e:
        print('Non-matching file found in data file: ', w)
        print(e)
        return 0


# TODO take this out -- for testing ONLY
#bvh_file = os.path.join('Splits', 'NaturalTalking_005', 'NaturalTalking_005_split_11_frame_1612_2115.bvh')
#wav_file = os.path.join('Splits', 'NaturalTalking_005', 'NaturalTalking_005_split_11_time_26.8667204_35.2500705.wav')
#txt_file = os.path.join('Splits', 'NaturalTalking_005', 'NaturalTalking_005_split_11_time_26.8667204_35.2500705.json')
#modat = get_positions(bvh_file)[0]