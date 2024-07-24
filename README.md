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
In the example folder is a short example on how to use the GUESSPy class. 
   
# Publications
Please cite following publications if you use this repository:

   * [GUESS](https://academic.oup.com/bioinformatics/article/35/14/2458/5216311?login=true)
   * [The Virtual Doctor](TODO)
   
# Authors
   * Jan Benedikt Ruhland, jan.ruhland@hhu.de
   
# License
MIT license (see license file). 
