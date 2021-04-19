from bvh_helpers import *
from local_modules.pymo.viz_tools import save_fig, draw_stickfigure, draw_stickfigure3d
import os
import matplotlib.pyplot as plt
import matplotlib.cm as cm

def get_num_frames(bvh_file):
    return 0


def bvh_to_mp4(bvh_file):
    modat = get_positions(bvh_file)[0]      # assume 1 file at a time.
    os.path.mkdir('tmp_figs')
    num_frames = get_num_frames(bvh_file)
    for i in range(num_frames):
        fig = draw_stickfigure3d(modat, frame=i)
        save_fig(fig)


def generate_video(img):
    for i in xrange(len(img)):
        plt.imshow(img[i], cmap=cm.Greys_r)
        plt.savefig('tmp_figs' + "/file%02d.png" % i)

    os.chdir("your_folder")
    subprocess.call([
        'ffmpeg', '-framerate', '8', '-i', 'file%02d.png', '-r', '30', '-pix_fmt', 'yuv420p',
        'video_name.mp4'
    ])
    for file_name in glob.glob("*.png"):
        os.remove(file_name)