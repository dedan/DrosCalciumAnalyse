
Scripts
=======

This folder contains some scripts that are used for data preprocessing or data *maintenance*. It is not really part of the analysis but helps to *prepare* the data.

* `split_micro`: perform vertical cut of original image and mask so that we can continue with the normal analysis tools (`plot_regions.py`, etc..)
* `turnscript`:  some of the data are flip along the horizontal axis. This script corrects this.
* `timeseries_from_png`: create time series objects from png images