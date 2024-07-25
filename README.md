# GUESSPY
This repository provides a python implementation of [GUESS](https://academic.oup.com/bioinformatics/article/35/14/2458/5216311?login=true). The repository is still under development, so use it at your own risk.

## The Calibration Procedure
GUESS calculates the probability of a class based on a machine learning score, thereby calibrating these scores. It uses log-likelihood optimization to fit a set of distributions to the likelihood distribution, estimating a continuous probability density function.

## Dependencies
The Code was tested with following packages:
   * Numpy Version 1.26.3
   * Pickle Version 4.0
   * Scipy Version 1.11.4
   
## Usage
If you are just interested in the calibrated score you have to provide the true classes (*y_classes*) and the machine learning scores (*x_scores*) to the GUESSPy class. If you want to see the analytical steps, set the verbose tag to *True*. You can then let the class calculate the calibrated score of an unknown score *x_i*. You always have to specify the number of bins *N* to be used for the analysis.

```
from GUESSPy import GUESSPy

g=GUESSPy(y_classes, x_scores, verbose=True)

x_i=0.1
calibrated_x_i=g.getCalibration(x_i, 20)
```
  
You can save the calibration results which enables you to use the calibration **without** providing data (this is important for sensitive data, e.g. medical data), using the saveState() method:

```
g.saveState("test.pkl")
g.loadState("test.pkl")
```

## Advanced Usage
You can check each analytic step seperately by, starting by the binning step:

```
binWidth, binCenters, binWeights, N_total, binMeans = g.getBins(20, g.c_0) 
```
 
You then can plot the binned scores together with the fit function which has to be fit to the data first and is saved in the class:

```
g.getBestLikelihood(c=0, N=20)
x=np.linspace(0,1,1000)
fitFunction=g.getLikelihood(x, c=0)
```

To compare the calibrated score with the original machine learning scores you can use either the expected calibration error (ECE), the maximum calibration error (MCE), or the classification error (CLE) which are implemented in the class.  

```
ECE_0=g.ECE(c=0, N=20, useClibrated=True)
MCE_0=g.MCE(c=0, N=20, useClibrated=True)
CLE_0=g.CLE(c=0, N=20, useClibrated=True)

```
 
# Publications
Please cite the original GUESS paper and the paper for which this repository was created:

   * [GUESS](https://academic.oup.com/bioinformatics/article/35/14/2458/5216311?login=true)
   * [The Virtual Doctor](TODO)
   
# Authors
   * Jan Benedikt Ruhland, jan.ruhland@hhu.de
   
# License
MIT license (see license file). 
