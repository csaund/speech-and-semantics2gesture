## WIP

from argparse import ArgumentParser
from local_modules.pymo.writers import BVHWriter
from caro_tests.bvh_helpers import get_positions


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
    f = open(params.output, 'w')
    W.write(modat, f)
    f.close()
