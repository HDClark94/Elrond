# Elrond: 
The ELectrophysiology Repository for Open-source Neural Data

## Overview
A comprehensive pipeline for electrophysiological recordings in rodents.

This repository contains scripts to call spike sorting and post processing of electrophysiological data recorded from rodents across a range of behavioural tasks including free-moving open field, head-fixed virtual environments, sleep and video playback experiments. Spike sorting is controlled primarily through SpikeInterface objects and functions, Curation is possible through Phy while post-processing is controlled through custom scripts.

The codebase can be used at every stage of research.
1. Data preparation and daily analysis to provide vital feedback to inform experimental decisions during an experiment
2. Post experimental anaylsis for projects and publication
3. Data collation for open-source publication of raw data using the DANDI Archieve in Neuro Data without Borders (NWB) formatting.


### Example output figures
![image](https://user-images.githubusercontent.com/16649631/123976239-e806cd80-d9b5-11eb-839b-28c86352e201.png)

## How to contribute
Please submit an issue to discuss.

## FAQ
Spike sorting crashed with a "too many files open" error: Look in additional_files/fix_file_open_limit_linux for instructions for a workaround
