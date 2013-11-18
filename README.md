#SimpleBool

SimpleBool is a python package for simulation and analysis of dynamic Boolean network models.

This software package was inspired by the pioneering work of István Albert and Réka Albert in biological Boolean networks .
SimpleBool is similar to their python package [Booleannet](http://code.google.com/p/booleannet/) but does not require coding experience. We used a pre-calculated truth table to update the state of each node and simplified the Boolean model representation. Therefore SimpleBool runs relatively faster than Booleannet. However, for advanced users, Booleannet may be recommended since it is more flexible and contains more complex updating methods such as piece wise differential method. 


##Features

1. Automatically running simulations of a Boolean network model and plotting results. 
2. Using model input file (containing Boolean rules) and parameter input file as inputs. No coding experience is required.
3. Easily performing systematic node perturbations studies. Both single and multiple perturbations are supported.
4. Constructing the state transition graph of a Boolean network model from all initial states or randomly chosen initial states.
5. Automatically identifying attractors and basins of attraction of a Boolean network model. 

##Requirements

SimpleBool works on Python 2.7. 

Matplotlib(http://matplotlib.org/) is required for BoolSimu.py to plot simulation results.

Numpy(http://www.numpy.org/) and networkX(http://networkx.github.io/) are required for BoolAttractor.py to construct state transition graph and identify attractors.

Enthought Python distribution or Enthought Canopy (https://www.enthought.com/) is highly recommended since they contain all the packages that SimpleBool needed.

##Installation

No particular installation steps are needed.

For Windows, just copy these scripts to your working directory where your model input file and simulation parameter file exits and execute the scripts.

For Linux, you can change the permission of the scripts to executable using 'chmod +X' and copy them to some directory that in your PATH environment variable.
Or you can also put them in your working directory containing the model input file and parameter input file and typing 'python [scrip name]' to execute the script.


##Quick Reference

###Model input file
A model input file is a text file that contains the Boolean rules for each node in the network. A model input file can be considered as the definition of a Boolean network model.

A simple model input file for a 4-node Boolean network model may look like:

```
A* = B or C
B* = A and D
C* = (A or B) and not D
D* = not B
```

'A','B','C','D' are the nodes' names in a Boolean network. In a particular biological network, a nodes can be a protein, a gene or a small molecule.

'and', 'or' and 'not' are logical operators that define the regulatory relationships between target node and its regulators. Parentheses can be used to denote the priority of the regulatory relationships. 

'A* = B or C' means the new value of A are determined by the old value of B and C.

A sample model input file of a Boolean network model describing the development of colitis-associated colon caner: [CAC.txt](https://github.com/lujunyan1118/SimpleBool/blob/master/examples/CAC.txt) 

###Running simulations and plotting using BoolSimu.py
For using BoolSimu.py to perform dynamic simulations on a Boolean network model and plot the simulation results, a parameter input file named 'simu.in' is needed to specify the simulation and plotting information.

The "simu.in" file looks like:
```
rules = CAC.txt
turn_on = DC
turn_off =
ini_on = APC,IKB
ini_off = Proliferation,Apoptosis
mode=sync
rounds = 500
steps = 30
plot_nodes = Apoptosis,Proliferation,IL6,TNFA,NFKB,STAT3
missing=random
```
**rules**: 'rules' specifies the model input file ('CAC.txt').

**turn_on**, **turn_off**: 'turn_on' and 'turn_off' each accepts a list of node names separated by ','. These two parameters specify the nodes that need to be **kept** in ON state or OFF state during simulation.

**ini_on**, **ini_off**: 'ini_on' and 'ini_off' each accepts a list of node names separated by ','. These two parameters specify the nodes whose states should be set in ON or OFF states **at the first step** of simulation.

**mode**: this parameter specifies the updating method. Three methods are implemented: 'Sync' (Synchronous updating method), 'GA' (General Asynchronous updating method) and 'ROA' (Random Order Asynchronous updating method). For the detailed explanation of these three methods, please consult the papers [Attractor analysis of asynchronous Boolean models of signal transduction networks](http://www.sciencedirect.com/science/article/pii/S0022519310003796) by Assieh Saadatpoura, István Albertb and Réka Albert.

**rounds**: 'rounds' specifies the rounds of simulation started from randomly selected initial states.

**steps**: 'steps' specified the steps of iteration during each round of simulation.

**plot_nodes**: this parameter specifies the nodes whose simulation results needed to be plotted when the simulation is done.

**missing**: this parameter specified the states of those nodes whose initial states are not determined from 'turn_on','turn_off','ini_on' and 'ini_off'. Values can be 'random', 'True', or 'False'.

* The parameter file, model input file should be in the same folder. 


###Performing node perturbation studies using BoolMutation.py

A parameter input file, named "mutation.in", is needed to perform the node perturbations using BoolMutation.py

The "mutation.in" file looks like:
```
rules = CAC.txt
turn_on = DC
turn_off =
ini_on = APC,DC
ini_off = Apoptosis,Proliferation
rounds = 500
steps = 100
mode=ROA
mutation_list=list_ON.txt,list_OFF.txt
keep_state=False,True
mutation_mode=single
observe_list=Proliferation,Apoptosis,STAT3,NFKB,BCATENIN
```

This first eight parameters are the same as 'simu.in' used by BoolSimu.py

**mutation_list**: 'mutation_list' specifies the files that contain the nodes that need to be perturbed. Sample input list of nodes: [list_ON.txt](https://github.com/lujunyan1118/SimpleBool/blob/master/examples/list_ON.txt),[list_OFF.txt](https://github.com/lujunyan1118/SimpleBool/blob/master/examples/list_OFF.txt).

**keep_state**: 'keep_state' specifies the perturbed state of the nodes in each nodes list provided by 'mutation_list' parameter, accordingly.

**mutation_mode**: this parameter denotes whether single or double perturbation study should be performed. 

**observe_list**: this parameter specifies the nodes whose final states should be written out in the output file.

* BoolMutation.py writes out a .csv file that contain the perturbed nodes and their states, as well as the states of the nodes specified in the 'observe_list'.

###Identifying attractors and basins of attraction using BoolAttractor.py

Similarly, a parameter input file, "steady.in", is needed by BoolAttractor.py

The "steady.in" file looks like:

```
rules = CAC.txt
turn_on =
turn_off =
ini_on = 
ini_off = Apoptosis,Proliferation
mode=Sync
steps=100000
rounds=100
initial_limit=0
```
This first eight parameters are the same as 'simu.in' used by BoolSimu.py

**initial_limit**: this parameter specifies the maximum number of initial states generated when constructing the state transition graph. "0" means no limitation and BoolAttractor.py this will use all the possible initial states to construct the state transition graph. For a Boolean network with n nodes, the number of states is 2^n. Therefore, for large Boolean networks, using all the possible initial states may be extremely slow or run into memory problems.  

* When finished, a folder named 'Data' containing the all result files is generated in the working directory.

* These results include the whole state transition graph ('**TransGraph.txt**'), the information of point attractors and cyclic attractors ('**Point_attractors.csv**' and '**Cyclic_attractors.csv**'), summarization of the basins of attractors and basin intersections ('**Summary_basin.txt**'), and the nodes that stabilized on a certain state in all the attractors ('**Fixed_nodes**').


##Authors

Junyan Lu  lujunyan_1118@aliyun.com.cn

Drug Discovery and Design Center, Shanghai Institute of Materia Medica, Chinese Academy of Sciences

Zhongjie Liang  zjliang@suda.edu.cn

Center for Systems Biology, Soochow University, 215006, China

##Copyrights
Copyright (C) 2013  Junyan Lu under [GPL3.0](https://github.com/lujunyan1118/SimpleBool/blob/master/LICENSE)
