SimpleBool
==========

A python package for simulation and analysis of dynamic Boolean network models


Features
--------
1. Automatically running simulations of a Boolean network model and ploting results. 
2. Using model input file (containing Boolean rules) and parameter input file as inputs. No coding experience is required.
3. Easily performing systematic node perturbations studies. Both single and multiple perturbations are supported.
4. Constructing the state transition graph of a Boolean network model from all initial states or randomly chosen initial states.
5. Automatically identifying attractos and basins of attraction of a Boolean network model. 

Requirements
------------
SimpleBool works on Python 2.7. 

Matplotlib(http://matplotlib.org/) is required for BoolSimu.py to plot simulation results.

Numpy(http://www.numpy.org/) and netowrkX(http://networkx.github.io/) are required for BoolAttractor.py to construct state transition graph and identify attractos.

Enthough Python distribution or Enthought Canopy (https://www.enthought.com/) is highly recommended since they contain all the packages that SimpleBool needed.

Installation
-------------
No particular installation steps are needed.

For Windows, just copy these scripts to your workding directory where your model input file and simulation parameter file exsits


