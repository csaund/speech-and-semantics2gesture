import numpy as np
import pandas as pd

# changed to win line endings
class BVHWriter():
    def __init__(self):
        pass

    def write(self, X, ofile, framerate=-1):

        # Writing the skeleton info
        ofile.write('HIERARCHY\r\n')

        self.motions_ = []
        self._printJoint(X, X.root_name, 0, ofile)

        # Writing the motion header
        ofile.write('MOTION\r\n')
        ofile.write('Frames: %d\r\n' % X.values.shape[0])

        if framerate > 0:
            ofile.write('Frame Time: %f\r\n' % float(1.0 / framerate))
        else:
            ofile.write('Frame Time: %f\r\n' % X.framerate)

        # Writing the data
        self.motions_ = np.asarray(self.motions_).T
        lines = [" ".join(item) for item in self.motions_.astype(str)]
        ofile.write("".join("%s\r\n" % l for l in lines))
        ofile.close()

    def _printJoint(self, X, joint, tab, ofile):

        if X.skeleton[joint]['parent'] == None:
            ofile.write('ROOT %s\r\n' % joint)
        elif len(X.skeleton[joint]['children']) > 0:
            ofile.write('%sJOINT %s\r\n' % ('\t' * (tab), joint))
        else:
            ofile.write('%sEnd site\r\n' % ('\t' * (tab)))

        ofile.write('%s{\r\n' % ('\t' * (tab)))

        ofile.write('%sOFFSET %3.5f %3.5f %3.5f\r\n' % ('\t' * (tab + 1),
                                                      X.skeleton[joint]['offsets'][0],
                                                      X.skeleton[joint]['offsets'][1],
                                                      X.skeleton[joint]['offsets'][2]))
        rot_order = X.skeleton[joint]['order']

        # print("rot_order = " + rot_order)
        channels = X.skeleton[joint]['channels']
        rot = [c for c in channels if ('rotation' in c)]
        pos = [c for c in channels if ('position' in c)]

        n_channels = len(rot) + len(pos)
        ch_str = ''
        if n_channels > 0:
            for ci in range(len(pos)):
                cn = pos[ci]
                self.motions_.append(np.asarray(X.values['%s_%s' % (joint, cn)].values))
                ch_str = ch_str + ' ' + cn
            for ci in range(len(rot)):
                try:
                    cn = '%srotation' % (rot_order[ci])
                    self.motions_.append(np.asarray(X.values['%s_%s' % (joint, cn)].values))
                    ch_str = ch_str + ' ' + cn
                except Exception as e:
                    print('Absolute hail mary cause this didnt exist: %s_%s' % (joint, cn))
                    cn = '%sposition' % (rot_order[ci])
                    self.motions_.append(np.asarray(X.values['%s_%s' % (joint, cn)].values))
                    ch_str = ch_str + ' ' + '%srotation' % (rot_order[ci])
        if len(X.skeleton[joint]['children']) > 0:
            # ch_str = ''.join(' %s'*n_channels%tuple(channels))
            ofile.write('%sCHANNELS %d%s\r\n' % ('\t' * (tab + 1), n_channels, ch_str))

            for c in X.skeleton[joint]['children']:
                self._printJoint(X, c, tab + 1, ofile)

        ofile.write('%s}\r\n' % ('\t' * (tab)))
