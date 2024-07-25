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
If you're only interested in obtaining the calibrated score, you need to supply the true classes (*y_classes*) and the machine learning scores (*x_scores*) to the GUESSPy class. To view the analytical steps, set the *verbose* tag to True. After that, you can have the class calculate the calibrated score for an unknown score *x_i*. Remember to always specify the number of bins *N* for the analysis.

```
from GUESSPy import GUESSPy

g=GUESSPy(y_classes, x_scores, verbose=True)

x_i=0.1
calibrated_x_i=g.getCalibration(x_i, 20)
```
  
You can save the calibration results using the saveState() method. This allows you to use the calibration **without** needing to provide the data again, which is crucial for handling sensitive information, such as medical data.

```
g.saveState("test.pkl")
g.loadState("test.pkl")
```

## Advanced Usage
You can review each analytical step individually, starting with the binning step:

```
binWidth, binCenters, binWeights, N_total, binMeans = g.getBins(20, g.c_0) 
```
 
You can then plot the binned scores along with the fit function, which must first be fitted to the data and is saved within the class.:

```
g.getBestLikelihood(c=0, N=20)
x=np.linspace(0,1,1000)
fitFunction=g.getLikelihood(x, c=0)
```

To compare the calibrated score with the original machine learning scores, you can use the expected calibration error (ECE), the maximum calibration error (MCE), or the classification error (CLE), all of which are implemented in the class.

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
