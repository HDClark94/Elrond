## Overview

Manual curation of the automated spikes sorter's output is sometimes required. An example of this is when all false positives need to be eliminated, for instance in an experiment where a small population of cells is opto-tagged.

You can use the pipeline to curate data by executing these steps:
- run pipeline
- edit settings for in manual_curation_settings.py to choose a recording you want to curate _(make sure to deploy if you're using a remote env but don't commit)_
- run manual_curation_pre_proces.py
- open spike sorting results in phy (made by the pre-processing script)
- curate in phy
- run manual_curation_post_process.py
- run pipeline again - it will use the curated output by default that manual_curation_post_process.py made


First, you will need set up phy on Eleanor

## Steps for setting up phy on Eleanor

1. make phy file using manual_curation_pre_process.py

2. Install phy on Eleanor following these instructions:

[https://github.com/cortex-lab/phy/](https://github.com/cortex-lab/phy/)

3. Run phy - [params.py](http://params.py) is the name of the file name by manual_curation.py

 phy template-gui [params.py](http://params.py/)
 
 if the above doesn't work, try this:
 
 QT_QPA_PLATFORM=xcb phy template-gui params.p

there will be some packages missing most likely. Install these too. You should eventually get an error message about not having a display or something like that at this stage. This is because you also need to set up the x-11 forwarding, so that's fine.

For setting up the X-11 forwarding:

you can use MobaXtermit's an ssh client like putty but it has a built in X11 server. (You can SSH to a linux VM and run glxgears to test that it works)

Download this program from here:

[https://mobaxterm.mobatek.net/download-home-edition.html](https://mobaxterm.mobatek.net/download-home-edition.html)

Log into ssh session (click on session, select private key in advanced ssh settings, set the user name to ubuntu and enter your pass phrase).

Test if the x-11 forwarding works by running 

xeyes

A pair of eyes that follow the cursor should pop up (after installing this program - it'll tell you how to install).

You can now try to run phy again:
![image](https://user-images.githubusercontent.com/16649631/127920284-8f33c5ae-08b5-4477-bc11-d7309769344e.png)

There is an error in the vesion of spikeinterface I used that doesn't allow setting the ms_before and ms_after parameters to a non-default value. To fix this, edit the spiketoolkit file on Eleanor (see path to file below) and add the missing parameters to the get_unit_amplitudes function (spikeinterface version 0.12.0 has this bug) - They merged my PR that fixes this so it might be good now if you update spikeinterface but I haven't checked.

**(**[https://github.com/SpikeInterface/spikeinterface/issues/191](https://github.com/SpikeInterface/spikeinterface/issues/191)**)**

:~/miniconda3/envs/env/lib/python3.6/site-packages/spiketoolkit/postprocessing$ vim postprocessing_tools.py

![image](https://user-images.githubusercontent.com/16649631/127920426-cbe09172-4551-40e7-b350-1ed4f5894cfb.png)


## Pre-process data and make phy file

- Copy data from server and paired recordings
- Make phy input file from concatenated data
- Save stitch points as csv file

(env) ubuntu@klara-frosty:~/manual/M3_2021-05-26_14-19-02_of$ cat stitch_points.csv
3.637964800000000000e+07
5.394944000000000000e+07
1.490216960000000000e+08

It takes a while to run, but I don't think it takes longer to run one relative to curating another so someone could work on this relatively continuously if they wanted to.

Warning: spikeinterface fills up the tmp folder completely. This needs to be emptied.

It's possible to open combined phy files and they look okay:
![image](https://user-images.githubusercontent.com/16649631/127920932-6592dea8-30d8-4dbe-9aef-1a3e550b3a7e.png)

## Manual curation in phy
The pre-processing script made input files for phy on Eleanor. You can find them here:
![image](https://user-images.githubusercontent.com/16649631/130420124-af030873-558c-43b5-b70d-c3db78f211ad.png)


Read these instructions, because the default views are unusable for our data and it explains how to do manual curation: [https://phy.readthedocs.io/en/latest/clustering/](https://phy.readthedocs.io/en/latest/clustering/)

![image](https://user-images.githubusercontent.com/16649631/127920995-58c13c6d-2e3e-40a6-9991-c89451a5d10e.png)


Suggestions for doing manual curation (but read detailed guidelines on phy website)

- Select/reset wizard
- If the clusters don't look like they need merging, press space, otherwise edit. Pressing space will keep going through all the pairs you need to check
- Merge (press g) when the waveforms look similar, the cross-correlation shows a refractory period and the firing rates correlate over time / look like the complement each other suggesting a drift. For bursty cells, the only difference might be the size of the amplitude and the cross-correlogram will help.

for example I think these should be merged because the PCA view suggests the red cluster is a segment of the blue one, the waveforms look very similar, there's a gap in the middle of the cross-correlogram (however this is not very meaningful because the red unit has very few spikes) and their firing times correlate across the recording (bottom right):

![image](https://user-images.githubusercontent.com/16649631/127921057-3644d9c4-3ba9-4ec8-8857-254017a41cba.png)

and this is noise:
![image](https://user-images.githubusercontent.com/16649631/127921096-8253853c-c473-4be6-9010-ab9aa0575e91.png)


Select all the clusters to double check the results after curation. I find the correlogram view very helpful, and it's also good to see the action potentials side by side:
![image](https://user-images.githubusercontent.com/16649631/127921172-dc5a6280-eb29-46f7-992d-455343650f41.png)

When you're done, make sure to set all the good clusters to 'good', multi-unit activity to 'MUA' and noise clusters to 'noise' (alt+g / alt + m / alt + n). This is important, because otherwise we won't know which ones are noise when we load them in python. It doesn't look like it's possible to just delete them.

phy should modify the files I highlighted and it'll also print out what you did on the terminal:
![image](https://user-images.githubusercontent.com/16649631/127921235-18aebecc-fe1f-4f0b-8301-c5612fd20213.png)
![image](https://user-images.githubusercontent.com/16649631/127921256-55726a84-424a-4aa7-9efc-bbf984ba82ba.png)

![image](https://user-images.githubusercontent.com/16649631/127921309-e13d1a8a-85cf-49fb-8155-9b245dd117e3.png)

Make sure it's saved correctly. (By clicking on save and exiting.) The cluster groups should be updated in the file below. This didn't happen when I just closed the program and clicked on save...

![image](https://user-images.githubusercontent.com/16649631/127921347-4a3b9539-6e2e-4213-9bbd-2f43a2b3fc2a.png)


## Process phy output

Run manual_curation_post_process.py. It will do the following things:

- load data (phy outputs and recording/stitch_points.csv from main recording folder)
- split for paired recordings
- convert to pipeline format (spatial_firing_curated.pkl)
- remove noise and mua clusters (save in another df) based on the cluster_group.tsv file
- add primary channel to df (based on waveforms) so the data is like the MS output
- upload to datastore
- delete local folder

todo: recalculate cluster qualities and compare to make sure we improved the data and warn the user if it got worse?? - I haven't done this

## Run pipeline
I modified the pipeline to check whether the curated output exists and use it if it does. 












