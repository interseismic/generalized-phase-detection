#! /bin/env python
# Automatic picking of seismic waves using Generalized Phase Detection
# See http://scedc.caltech.edu/research-tools/deeplearning.html for more info
#
# Ross et al. (2018), Generalized Seismic Phase Detection with Deep Learning,
#                     Bull. Seismol. Soc. Am., doi:10.1785/0120180080
#
# Author: Zachary E. Ross (2018)
# Contact: zross@gps.caltech.edu
# Website: http://www.seismolab.caltech.edu/ross_z.html

import string
import time
import argparse as ap
import sys
import os

from gpd.gpd import gpd
from gpd.gpd_predict_params import gpd_predict_params
import obspy.core as oc

# -------------------------------------------------------------

if __name__ == "__main__":
    parser = ap.ArgumentParser(
        prog='gpd_predict.py',
        description='Automatic picking of seismic waves using'
                    'Generalized Phase Detection')
    parser.add_argument(
        '-I',
        type=str,
        default=None,
        help='Input file')
    parser.add_argument(
        '-O',
        type=str,
        default=None,
        help='Output file')
    parser.add_argument(
        '-P',
        default=True,
        action='store_false',
        help='Suppress plotting output')
    parser.add_argument(
        '-V',
        default=False,
        action='store_true',
        help='verbose')
    args = parser.parse_args()

    # Reading in input file
    fdir = []
    evid = []
    staid = []
    with open(args.I) as f:
        for line in f:
            tmp = line.split()
            fdir.append([tmp[0], tmp[1], tmp[2]])
    nsta = len(fdir)

    my_gpd = gpd('model_pol.json', "model_pol_best.hdf5", n_gpu=0)

    ofile = open(args.O, 'w')

    for i in range(nsta):
        fname = fdir[i][0].split("/")
        if not os.path.isfile(fdir[i][0]):
            print("%s doesn't exist, skipping" % fdir[i][0])
            continue
        if not os.path.isfile(fdir[i][1]):
            print("%s doesn't exist, skipping" % fdir[i][1])
            continue
        if not os.path.isfile(fdir[i][2]):
            print("%s doesn't exist, skipping" % fdir[i][2])
            continue
        st = oc.Stream()
        st += oc.read(fdir[i][0])
        st += oc.read(fdir[i][1])
        st += oc.read(fdir[i][2])

        params = gpd_predict_params(filter_data=True, decimate_data=False,  min_proba=0.95,
                                    freq_min=3.0, freq_max=20.0, n_shift=10, half_dur=2.00,
                                    only_dt=0.01,  batch_size=1000*3, verbose=args.V,
                                    plot=args.P)

        phase_picks = my_gpd.predict_stream(st, params)
        for phase in phase_picks:
            picks = phase_picks[phase]
            for pick in picks:
                ofile.write("%s %s %s %s\n" % (picks[0], picks[1], phase, picks[2].isoformat()))
    ofile.close()
