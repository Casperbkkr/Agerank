#AgeRank
##General info
The aim of this project has been to model the spread of airborne pathogens in a network based on demographic  data.
It consists of two major parts. The first being the creation of the network. Using the Polymod study and data about the makeup of households. 
Another important goal has been to make it easy to use different datasets.
##Requirements
* Bokeh
* Pandas
* Numpy
##How to use
###Inputs
There are two different types of inputs needed. The first are the parameters. 
These can be changed in the parameters.py file. Further explanation is included there. \
The second type are the datasets. There are standard datasets included. These datasets are about dutch demographics.
To change the datasets add them to the datafiles folder and change the path for the corresponding filename.
To run the code you need to specify the timesteps to be simulated and the number of times the model is to be ran for all vaccination strategies.
The results will be placed in the results directory. These can be visualized with the show_plots.py file.

