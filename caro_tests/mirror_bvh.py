## WIP

from argparse import ArgumentParser
from local_modules.pymo.writers import BVHWriter
from caro_tests.bvh_helpers import get_positions
import numpy as np
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

from local_modules.pymo.parsers import BVHParser
from local_modules.pymo.data import Joint, MocapData
from local_modules.pymo.preprocessing import *
from local_modules.pymo.writers import *

## to be run BEFORE splitting the bvh files, I think.


def mirror_downsample_bvh(bvh_dir, fps):
    print('mirroring bvh')
    bvh_files = os.listdir(bvh_dir)
    bvh_files = [f for f in bvh_files if '.bvh' in f]

    full_filenames = []
    for f in bvh_files:
        ff = ''
        if f.endswith('.bvh'):
            ff = os.path.join(bvh_dir, f)
        else:
            ff = os.path.join(bvh_dir, f + '.bvh')
        full_filenames.append(ff)

    out_data, data_pipeline = extract_joint_angles(full_filenames, fps=fps)
    print('writing bvh')
    write_bvh(data_pipeline, out_data, full_filenames, fps=fps)


# Get the joint angles for all bvh files in the dir.
def extract_joint_angles(bvh_files, fps):
    p = BVHParser()

    data_all = []
    for f in bvh_files:
        try:
            data_all.append(p.parse(f))
        except:
            print('COULD NOT PARSE BVH FILE: ', f)
            print('skipping...')
            continue

    data_pipe = Pipeline([
       ('dwnsampl', DownSampler(tgt_fps=fps,  keep_all=False)),
       ('root', RootTransformer('hip_centric')),
       ('mir', Mirror(axis='X', append=False)),
       ('jtsel', JointSelector(['Spine','Spine1','Spine2','Spine3','Neck','Neck1','Head','RightShoulder', 'RightArm', 'RightForeArm', 'RightHand', 'LeftShoulder', 'LeftArm', 'LeftForeArm', 'LeftHand'], include_root=True)),
       ('exp', MocapParameterizer('expmap')),
       ('cnst', ConstantsRemover()),
       ('np', Numpyfier())
    ])

    out_data = data_pipe.fit_transform(data_all)
    return out_data, data_pipe


# write all bvh files to the output dir
def write_bvh(data_pipeline, out_data, filenames, fps):
    inv_data = data_pipeline.inverse_transform(out_data)
    writer = BVHWriter()
    for i in range(0, out_data.shape[0]):
        try:
            with open(filenames[i], "w") as f:
                writer.write(inv_data[i], f, framerate=fps)
        except Exception as e:
            print('unable to print bvh file %s / (%s, %s) ' % (i, len(filenames), len(inv_data)))
            print(e)


if __name__ == "__main__":
    # Setup parameter parser
    parser = ArgumentParser()
    parser.add_argument('--bvh_dir', '-orig', default="../../dataset/raw_data/Motion",
                                   help="Path where original motion files (in BVH format) are stored")

    params = parser.parse_args()

    files = []
    # Go over all BVH files
    for r, d, f in os.walk(params.bvh_dir):
        for file in f:
            if '.bvh' in file:
                ff = os.path.join(r, file)
                #print(ff)
                # basename = os.path.splitext(os.path.basename(ff))[0]
                files.append(ff)

    #print(files)
    out_data, data_pipeline = extract_joint_angles(files, fps=20)
    write_bvh(data_pipeline, out_data, files, fps=20)