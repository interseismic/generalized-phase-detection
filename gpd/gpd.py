import os
import numpy as np
import obspy.core as oc
import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv1D, MaxPooling1D
from keras import losses
from keras.models import model_from_json

from gpd.utils import sliding_window


class gpd:
    """Automatic picking of seismic waves using Generalized Phase Detection
    See http://scedc.caltech.edu/research-tools/deeplearning.html for more info
    
    Author: Zachary E. Ross (2018)                
    Contact: zross@gps.caltech.edu                        
    Website: http://www.seismolab.caltech.edu/ross_z.html

    Ross et al. (2018), Generalized Seismic Phase Detection with Deep Learning,
        Bull. Seismol. Soc. Am., doi:10.1785/0120180080
    """

    def __init__(self, model_json_path, weights_hdf5_path, n_gpu=3):
        """Create a new Generalized Phase Detector using the specified model and weights.

        Args:
            model_json_path (str): path to the model definition file in JSON format.
            weights_hdf5_path (str): path to the file containing the model weights in HDF5 format.
            n_gpu (int): The number of GPUs to use.

        Attributes:
            model (a Keras model): The model defined in the passed in JSON, with the weights from the HDF5 file loaded into it.
        """
        if os.path.exists(model_json_path) == False:
            raise FileNotFoundError(
                "The model's JSON definition file could not be found at: "+model_json_path)

        if os.path.exists(weights_hdf5_path) == False:
            raise FileNotFoundError(
                "The model's HDF5 weights file could not be found at: "+weights_hdf5_path)

        json_file = open(model_json_path, 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        self.model = model_from_json(
            loaded_model_json, custom_objects={'tf': tf})

        # load weights into new model
        self.model.load_weights(weights_hdf5_path)
        if n_gpu > 1:
            from keras.utils import multi_gpu_model
            model = multi_gpu_model(model, gpus=n_gpu)
        print("Loaded model from disk")

    def predict_stream(self, st, params):
        """Predict phase labels for the passed in Stream

        Args:
            st (obspy.core.stream.Stream): An Obspy Stream instance containing three Traces (N-E-Z components).
            params (gpd_predict_params): Parameters that control preprocessing, hyperparameters, and any subsequent post-processing.

        Returns:
            A python dictionary with keys "P" and "S". Value of each key is a list of pick times (Obspy UTCDateTime instances)
        """
        picks = {"P": [], "S": []}
        latest_start = np.max([x.stats.starttime for x in st])
        earliest_stop = np.min([x.stats.endtime for x in st])
        st.trim(latest_start, earliest_stop)

        st.detrend(type='linear')
        if params.filter_data:
            st.filter(type='bandpass', freqmin=params.freq_min,
                      freqmax=params.freq_max)
        if params.decimate_data:
            st.interpolate(100.0)
        chan = st[0].stats.channel
        sr = st[0].stats.sampling_rate

        dt = st[0].stats.delta
        net = st[0].stats.network
        sta = st[0].stats.station

        if params.verbose:
            print("Reshaping data matrix for sliding window")
        tt = (np.arange(0, st[0].data.size,
                        params.n_shift) + params.n_win) * dt
        tt_i = np.arange(0, st[0].data.size, params.n_shift) + params.n_feat
        sliding_N = sliding_window(
            st[0].data, params.n_feat, stepsize=params.n_shift)
        sliding_E = sliding_window(
            st[1].data, params.n_feat, stepsize=params.n_shift)
        sliding_Z = sliding_window(
            st[2].data, params.n_feat, stepsize=params.n_shift)
        tr_win = np.zeros((sliding_N.shape[0], params.n_feat, 3))
        tr_win[:, :, 0] = sliding_N
        tr_win[:, :, 1] = sliding_E
        tr_win[:, :, 2] = sliding_Z
        tr_win = tr_win / np.max(np.abs(tr_win), axis=(1, 2))[:, None, None]
        tt = tt[:tr_win.shape[0]]
        tt_i = tt_i[:tr_win.shape[0]]

        ts = self.model.predict(
            tr_win, verbose=params.verbose, batch_size=params.batch_size)

        prob_S = ts[:, 1]
        prob_P = ts[:, 0]
        prob_N = ts[:, 2]

        from obspy.signal.trigger import trigger_onset
        trigs = trigger_onset(prob_P, params.min_proba, 0.1)
        p_picks = []
        s_picks = []
        for trig in trigs:
            if trig[1] == trig[0]:
                continue
            pick = np.argmax(ts[trig[0]:trig[1], 0])+trig[0]
            stamp_pick = st[0].stats.starttime + tt[pick]
            picks["P"].append([net, sta, stamp_pick])
            if params.verbose:
                print(f"{net} {sta} P {stamp_pick.isoformat()}")

        trigs = trigger_onset(prob_S, params.min_proba, 0.1)
        for trig in trigs:
            if trig[1] == trig[0]:
                continue
            pick = np.argmax(ts[trig[0]:trig[1], 1])+trig[0]
            stamp_pick = st[0].stats.starttime + tt[pick]
            picks["S"].append([net, sta, stamp_pick])
            if params.verbose:
                print(f"{net} {sta} S {stamp_pick.isoformat()}")

        if params.plot:
            import matplotlib as mpl
            import pylab as plt
            mpl.rcParams['pdf.fonttype'] = 42
            fig = plt.figure(figsize=(8, 12))
            ax = []
            ax.append(fig.add_subplot(4, 1, 1))
            ax.append(fig.add_subplot(4, 1, 2, sharex=ax[0], sharey=ax[0]))
            ax.append(fig.add_subplot(4, 1, 3, sharex=ax[0], sharey=ax[0]))
            ax.append(fig.add_subplot(4, 1, 4, sharex=ax[0]))
            for i in range(3):
                ax[i].plot(np.arange(st[i].data.size)*dt, st[i].data, c='k',
                           lw=0.5)
            ax[3].plot(tt, ts[:, 0], c='r', lw=0.5)
            ax[3].plot(tt, ts[:, 1], c='b', lw=0.5)
            for p_pick in p_picks:
                for i in range(3):
                    ax[i].axvline(p_pick-st[0].stats.starttime, c='r', lw=0.5)
            for s_pick in s_picks:
                for i in range(3):
                    ax[i].axvline(s_pick-st[0].stats.starttime, c='b', lw=0.5)
            plt.tight_layout()
            if params.plot_filename is not None:
                plt.savefig(params.plot_filename)
                if params.verbose:
                    print(f"Wrote plot to {params.plot_filename}")
            else:
                plt.show()

        return picks
