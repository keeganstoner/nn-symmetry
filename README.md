# Symmetry-via-Duality: Invariant Neural Network Densities from Parameter-Space Correlators

<b>Anindita Maiti, Keegan Stoner, and Jim Halverson</b>

<i>Northeastern University</i>


Code associated with the paper arxiv.org/abs/2106.xxxx.pdf

## Accuracy and Symmetry Breaking

To reproduce the symmetry breaking plots in Fig. 1, showing the two paramters mu and k affect training accuracy, first generate the data by running

```symmetry_breaking_paramters.py```

with options 

``` --run=[int]``` to repeat experiments with the same parameters. default = 0

``` --k=[int, 0 to 10]``` so that k of the last linear weights will have mean mw (the rest will have mean 0). default = 10

``` --mw=[float]``` to set the value of the mean for k of the parameters. If you want ALL parameters to have this mean, set k = 10. default = 0.0

``` --targets=["hot" or "cold"]``` to specify the target encoding. default = "hot" onehot


One can then run ```plot_heatmap.py``` to generate a heatmap given the ranges of parameters run from the previous script. To change the target encoding simply changed the commented line for the variable ```targets```. 

To see the effect of mw on accuracy only, such as in the one-cold plot of Fig. 1, run ```plot_mw.py```. 


## SO(5) invariance of n-pt functions

As an example of SO(D) invariance of the n-pt functions, we give the code for SO(5) invariance for the 2-pt and 4-pt functions. A demonstration of invariance for other D can be generated similarly with some small changes. In the npt_symmetry directory, run ```generate_models.py``` which has arguments

``` --width=[int]``` to specify the widths of the generated networks. default = 1000

``` --d-out=[int]``` to specify the output dimension of the networks. default = 5

Once the mdoels with ```d-out = 5``` are generated at a variety of widths, e.g. ```[5, 10, 50, 100, 1000]```, run ```npt.py``` for each width. This will save the 2pt and 4pt function tensors, as well as their statistical errors. 

Then to test for SO(5) invariance of the 2- and 4-pt functions, run all cells in ```so5_sym.ipynb```. 


# Code Authors

**Keegan Stoner**

**Anindita Maiti** ([aninditamaiti](https://github.com/aninditamaiti))

