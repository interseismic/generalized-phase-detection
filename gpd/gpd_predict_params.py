class gpd_predict_params:
    """An encapsulation of the parameters that control preprocessing, hyperparameters, and any subsequent post-processing during phase prediction.
    """

    def __init__(self,
                 filter_data=True, decimate_data=False,  min_proba=0.95,
                 freq_min=3.0, freq_max=20.0, n_shift=10, half_dur=2.00,
                 only_dt=0.01,  batch_size=1000*3, verbose=True,
                 plot=False, plot_filename=None):
                """Create a new instance of parameters to use during phase detection.

                Args:
                    filter_data (boolean): Whether to apply filtering to the data before predicting phase labels.
                    decimate_data (boolean): Whether to modify the data to be 100 Hz.
                    min_proba (float): The minimum softmax probability for phase detection
                    freq_min (float): If filtering is applied (filter_data = True), this is the lower frequency bound
                    freq_max (float): If filtering is applied (filter_data = True), this is the upper frequency bound
                    n_shift (int): The number of samples to shift when creating sliding windows of data.
                    half_dur (float):
                    only_dt (float):
                    batch_size (int): The number of samples to predict at a time.
                    verbose (boolean): Whether to output status messages as processing occurs.
                    plot (boolean): Whether to display a plot of the waveform with predicted phase overlayed
                    plot_filename (str): The name of the file to save the plot to (ignored if plot = False).

                Attributes:
                    n_win (int):
                    n_feat (int):
                """
                self.filter_data = filter_data
                self.decimate_data = decimate_data
                self.min_proba = min_proba
                self.freq_min = freq_min
                self.freq_max = freq_max
                self.n_shift = n_shift
                self.half_dur = half_dur
                self.only_dt = only_dt
                self.batch_size = batch_size
                self.verbose = verbose
                self.plot = plot
                self.plot_filename = plot_filename

                self.n_win = int(half_dur/only_dt)
                self.n_feat = 2*self.n_win
