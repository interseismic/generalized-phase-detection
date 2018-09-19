# Generalized Seismic Phase Detection with Deep Learning

This is a simple implementation of the Generalized Phase Detection (GPD) framework for seismic phase detection with deep learning. GPD uses deep convolutional networks to learn generalized representations of millions of P-wave, S-wave, and noise seismograms that can be used for phase detection and picking. The framework is described in
```
Ross, Z. E., Meier, M.-A., Hauksson, E., and T. H. Heaton (2018). Generalized Seismic Phase Detection with Deep Learning, Bull. Seismol. Soc. Am., doi: 10.1785/0120180080 [arXiv:1805.01075]
```
We have provided training datasets consisting of millions of seismograms at the Southern California Earthquake Data Center <http://scedc.caltech.edu/research-tools/deeplearning.html>. We strongly encourage others to download these datasets and improve upon our model architecture as described in the paper. If you do, let us know and we will include it so that others can use it with GPD.

## Requirements
The models provided here require keras and tensorflow. It is recommended to use GPUs because the computations are very heavy. You will need the two model files included here in order to run the script. These modules are changing all the time, and it may potentially lead to version conflicts. Let us know if this is the case and we can provide a more up-to-date version of the model.

## The gpd_predict script
This script is a very simple implementation of GPD for detection and picking of seismic waves in continuous data. You can run it the test data as follows,
```
./gpd_predict.py -V -I anza2016.in -O anza2016.out
```
This will write a set of phase picks for the 2016 Mw 5.2 Anza, California sequence to a file anza2016.out. It will also produce a plot of the three component traces, plot the picks, and the P- and S-wave probability streams.

The input file, specified with the -I flag, has an arbitrary number of rows and three columns. The columns correspond to the North, East, and Vertical channel filenames for a given chunk of 3-c data. Each row corresponds to the data for a different station. In the example input file, there is just one row, which means only a single station will be processed. The files can be of arbitrary duration. All three components are necessary because GPD is a three-component detection framework.

gpd_predict has a few hyperparameters that can be adjusted to suit your goals. They are all documented in the script at the top. Please see the BSSA paper to understand how the min_proba parameter works.

The model is trained assuming the data are 100 Hz sampling rate. I have no idea what will happen if they are not 100 Hz, but the script includes a flag to interpolate the data should you not want to experiment with these of the box scenarios. If you do want to experiment with this, let me know what you find. 

```
