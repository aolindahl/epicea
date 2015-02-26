# epicea
Code for the EPICEA experiment at the PLEIADES beamline at the synchrotron SOLEIL. So far there are two parts, a general *epicea_hdf5.py* data handler, and the experiment specific *first_look.py* script that produces some figures.

## epicea_hdf5.py
The module takes files exported from igor into textfiles. At the moment *events*, *ions* and *electrons* files are used. The data is read and stored into an hdf5 file which can then be acessed in a faster way.

## first_look.py
Produces some first views of the datasets of a specific experiment.
