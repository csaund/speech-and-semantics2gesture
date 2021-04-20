# Copyright 2020 by Patrik Jonell.
# All rights reserved.
# This file is part of the GENEA visualizer,
# and is released under the GPLv3 License. Please see the LICENSE
# file that should have been included as part of this package.


import requests
from pathlib import Path
import time
import os
from tqdm import tqdm
import argparse

# example usage
#  python -m caro_tests.bvh_to_mp4_batch --bvh_dir Splits/NaturalTalking_008-sentence/


def bvh_to_video(bvh_file, output, server_url):
    headers = {"Authorization": "Bearer j7HgTkwt24yKWfHPpFG3eoydJK6syAsz"}
    render_request = requests.post(
        f"{server_url}/render",
        files={"file": (bvh_file.name, bvh_file.open())},
        headers=headers,
    )
    job_uri = render_request.text

    done = False
    while not done:
        resp = requests.get(server_url + job_uri, headers=headers)
        resp.raise_for_status()

        response = resp.json()

        if response["state"] == "PENDING":
            jobs_in_queue = response["result"]["jobs_in_queue"]
            print(f"pending.. {jobs_in_queue} jobs currently in queue")

        elif response["state"] == "PROCESSING":
            print("Processing the file (this can take a while depending on file size)")

        elif response["state"] == "RENDERING":
            current = response["result"]["current"]
            total = response["result"]["total"]
            print(f"currently rendering, {current}/{total} done")

        elif response["state"] == "SUCCESS":
            file_url = response["result"]
            done = True
            break

        elif response["state"] == "FAILURE":
            raise Exception(response["result"])
        else:
            print(response)
            raise Exception("should not happen..")
        time.sleep(10)

    video = requests.get(server_url + file_url, headers=headers).content
    output.write_bytes(video)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--bvh_file", default=None, type=Path)
    parser.add_argument("--server_url", default="http://localhost:5001")
    parser.add_argument("--output", default=None, type=Path)
    parser.add_argument("--bvh_dir", default="",
                        help="directory of bvh files to attempt to convert")

    args = parser.parse_args()
    print(args)
    bvh_file = args.bvh_file
    bvh_dir = args.bvh_dir

    if bvh_file:
        output = args.output if args.output else bvh_file.with_suffix(".mp4")
        bvh_to_video(bvh_file, output, server_url=args.server_url)

    elif bvh_dir:
        fs = [f for f in os.listdir(bvh_dir) if f.endswith('resampled.bvh')]     # just test this for now!!
        for f in tqdm(fs):
            full_path = os.path.join(bvh_dir, f)
            output = Path(full_path).with_suffix(".mp4")
            print(output)
            bvh_to_video(Path(full_path), output, server_url=args.server_url)