# monotonic_fit
The purpose of this code is to, given a reference/wildtype/backbone sequence, 
and measurements of all the single mutation effects, transform the data in a 
manner that maximizes the additivity of these mutations to predict the effect
of new mutations. This idea of adding up mutations is very old (Wells, J. A. (1990). Additivity of mutational effects in proteins. Biochemistry 29, 8509–8517.). However, it is only more recently that interest has been in trying to optimize the monotonic transformation that allows mutations to be as additive as possible. If you use this code,
please cite my manuscript "Epistasis in a Fitness Landscape Defined by Antibody-Antigen Binding Free Energy" (https://www.cell.com/cell-systems/fulltext/S2405-4712(18)30478-2).

To run this program, I used python 3.6.6 installed from anaconda. This means that numpy, scipy, and matplotlib should be installed. Additionally, I made use of cvxopt as the engine to optimize the transformation. You can obtain this at:

```
#monotonic fit requirement
conda install -y -c conda-forge cvxopt
```

The engine of this code is monotonic_fit.py, and an example notebook can be accessed at monotonic_fit_example.ipynb. make_A.py is a helper script that converts sequence data into a sparse boolean matrix for fast linear algebra operations. In general this code requires you to specify the measured values of all single mutations and performs cross validation to determine the best smoothing levels (linearity). 
