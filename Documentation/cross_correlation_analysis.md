## Cross-correlation analysis

This application plots cross-correlograms between spike trains of pairs of neurons within a tetrode. To use it, edit PostSorting/CrossCorrelationAnalysis/plot_cross_correlograms.py and change the folder path to the data store folder you'd like to analyse. It will analyse all recordings inside the main folder, or you can run it on a single recording folder. (See more instructions in the comments in the script.)

Example plot:

![image](https://user-images.githubusercontent.com/16649631/177530345-a094f038-ea8d-4192-afb0-8868c022d297.png)

The plots include all combinations including autocorrelograms (diagonal). The x axis is in seconds. You can change the window size and bin size of the plots by editing the parameters of the function.
