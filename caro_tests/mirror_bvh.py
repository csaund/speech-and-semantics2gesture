## WIP

from argparse import ArgumentParser
from local_modules.pymo.writers import BVHWriter
from caro_tests.bvh_helpers import get_positions
import numpy as np

"""
def mirror_sequence(sequence):
    mirrored_rotations = sequence[:, 1:, :]
    mirrored_trajectory = np.expand_dims(sequence[:, 0, :], axis=1)

    temp = mirrored_rotations

    # Flip left/right joints
    mirrored_rotations[:, joints_left] = temp[:, joints_right]
    mirrored_rotations[:, joints_right] = temp[:, joints_left]

    mirrored_rotations[:, :, [1, 2]] *= -1
    mirrored_trajectory[:, :, 0] *= -1

    mirrored_sequence = np.concatenate((mirrored_trajectory, mirrored_rotations), axis=1)

    return mirrored_sequence
"""

if __name__ == "__main__":
    # Setup parameter parser
    parser = ArgumentParser()
    parser.add_argument('--file_name', '-bvh', default="NaturalTalking_010",
                        help="bvh file to extract")#
    parser.add_argument('--output', '-o', default="NaturalTalking_010",
                        help="where the output bvh lives")

    params = parser.parse_args()
    bvh_name = params.file_name

    print('getting motion data')
    modat = get_positions(bvh_name)[0]      ## the 0 here is because we only operate on 1 file at a time

    W = BVHWriter()
    print('writer: ', W)

    f = open(params.output, 'w')
    W.write(modat, f)
    f.close()
