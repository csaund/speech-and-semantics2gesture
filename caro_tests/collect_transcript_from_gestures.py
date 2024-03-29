from argparse import ArgumentParser
import os
import json

## from a folder and a filename, go through all json files (transcripts)
# and make a file in that folder that is JUST the transcripts, line by line.

# example usage
# python -m caro_tests.collect_transcript_from_gestures --data_dir Splits/NaturalTalking_008-sentence --output_file transcripts.txt

def sorter(w):
    try:
        n = w.split('_')[-4]
        return int(n)
    except Exception as e:
        print('Non-matching file found in data file: ', w)
        return 0


def collect_transcript_from_dir(data_dir, output):
    transcripts = os.path.join(os.getcwd(), data_dir, output)
    fns = os.listdir(data_dir)
    fns.sort(key=sorter)
    t_file = open(transcripts, 'w')
    i = 0
    for f in fns:
        if f.endswith('.json'):
            with open(os.path.join(os.getcwd(), data_dir, f)) as j_f:
                data = json.load(j_f)
                t = ""
                try:
                    for el in data['transcript']:
                        t += el + " "
                    t_file.write(t)
                    t_file.write("\n")
                except Exception as e:
                    print('could not get transcript for file ', f, ':', e)
            i += 1
    t_file.close()


if __name__ == "__main__":
    # Setup parameter parser
    parser = ArgumentParser()
    parser.add_argument('--data_dir', '-orig', default="./",
                                   help="Path where original motion files (in BVH format) are stored")
    parser.add_argument('--output_file', '-o', default="../utils/",
                        help="Path where the motion data processing pipeline will be stored")

    params = parser.parse_args()

    collect_transcript_from_dir(params.data_dir, params.output_file)
