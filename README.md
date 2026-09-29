<<<<<<< HEAD
﻿# SE-801 FMCW project

This repository contains code for the four-class, parent-grouped radar
spectrogram project and the evaluated S2 CNN track. S1, S3 and S4 results are
not part of the current experiment. The frozen outer folds use seed 2026.

Python acquisition and preprocessing scripts are included. Cluster job and
transfer scripts require individual credential review before publication.
The 77 GHz radar
source `.npy`, generated images, and trained `.pt` checkpoints are kept in
separate data storage and are not uploaded to GitHub by this repository.

For the editable manuscript, compile `overleaf/main.tex` with its `figures/`
directory. Atlas batch jobs use the local project/data directory configured
in the scripts. The S2 paper identifies outstanding protocol deviations;
do not interpret the current scripts as a completed four-track study.
