# Elrond: A comprehensive pipeline for electrophysiological recordings in rodents

## Overview

This repository contains scripts to call spike sorting and post processing of electrophysiological data recorded from rodents across a range of behavioural tasks.

The core pipeline runs on a linux computer with the aim of minimizing user interaction as much as possible, to provide a first pass analysis from which further more detailed analysis can be completed

For published work using this pipeline (or older versions thereof), subdirectories in /published provided contained code from each project

Please see more detailed documentation in the /documentation folder.

### Example output figures
![image](https://user-images.githubusercontent.com/16649631/123976239-e806cd80-d9b5-11eb-839b-28c86352e201.png)

## How to contribute
Please submit an issue to discuss.

## FAQ
Spike sorting crashed with a "too many files open" error: Look in additional_files/fix_file_open_limit_linux for instructions for a workaround
