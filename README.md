# Medical-Record-Linkage-Ensemble

This repository contains a replication study for the paper "Statistical supervised meta-ensemble algorithm for data linkage", published in Journal of Biomedical Informatics. The original Github repo is located https://github.com/ePBRN/Medical-Record-Linkage-Ensemble. 

Original paper citation:  
Kha Vo, Jitendra Jonnagaddala, Siaw-Teng Liaw,  
Journal of Biomedical Informatic, Volume 95, 2019,  
103220, ISSN 1532-0464  
https://doi.org/10.1016/j.jbi.2019.103220  


Paper Authors: 
Kha Vo <kha.vo@unsw.edu.au>,
Jitendra Jonnagaddala <jitendra.jonnagaddala@unsw.edu.au>,
Siaw-Teng Liaw <siaw@unsw.edu.au>.

Replication Study Authors:
Alec Mori <ajmori2@illinois.edu>
Andrew Sciotti <sciotti2@illinois.edu>


## Setup
All of the code in this repository used Python 3.6 or higher with these base prerequisite packages: `numpy`, `pandas`, `sklearn`, and `recordlinkage`. Other auxiliary packages are used for convenience. To install a missing package, use command `pip install package-name` in a terminal (i.e., cmd in Windows, or Terminal in MacOS).

Please install your python environment either via `pip install -r windows_requirements.txt` or using the `Makefile` which handles virtualenv construction and installation via pip. We have provided a windows specific requirements file, if you are on OS/Linux, please manually install the base prerequisite packages and then the remaining auxiliary ones.

## How to replicate
1. Prepare the cleaned datasets for Scheme A, which are stored in two files `febrl_UNSW_train.csv` and `febrl_UNSW_test.csv`. To reproduce those two files, run `gen_raw_febrl_datasets.py`.

2. Prepare the synthetic ePBRN-error-simulated datasets for Scheme B, which are stored in two files `ePBRN_dup_train.csv` and `ePBRN_dup_test.csv`. The original FEBRL dataset (all original, no duplicate) is contained in 2 files: `/USNW Error Generation/ePBRN_Datasets/ePBRN_D_original.csv` (test set) and `/USNW Error Generation/ePBRN_Datasets/ePBRN_F_original.csv` (train set). To reproduce `ePBRN_dup_train.csv` and `ePBRN_dup_test.csv`, run `Error_Generator.ipynb`. In the first cell of the notebook, change variable `inputfile` to either `ePBRN_D_original` or `ePBRN_F_original`, which is respectively corresponding to variable `outputfile` of `ePBRN_dup_test` or `ePBRN_dup_train`. 

3. Reproduce results for Scheme A in the paper by running all cells in `FEBRL_UNSW_Linkage.ipynb`.

4. Reproduce results for Scheme B in the paper by running all cells in `ePBRN_UNSW_Linkage.ipynb`.

5. The plots in the paper can be reproduced by running `Plots.ipynb`.


## Ablation study
To reproduce the error-augmented data ablation study run all cells in `ePBRN_USNW_Linkage_Ablation.ipynb`, this notebook includes the plotting portions as well.

## Computation complexity
If you would like to replicate the computation and memory requirements, please refer to `computation_checks/README.txt`. 

## File/folder details
Non-exhaustive list of files/folders:

`computation_checks`: Contains scripts and duplicated source code to running the computational complexity results collected in the replication study

`UNSW Error Generator`: Scripts and original data to produce Scheme B datasets (FEBRL + ePBRN derived data)

`gen_raw_febrl_datasets.py`: Script to generate `febrl_UNSW_{train|test}.py` for Scheme A.

`Makefile` + `windows_requirements.txt`: Helper makefile and list of python packages to replicate the python environment needed to run code

`training_utils.py`: Contains common training/evaluation utils used across Scheme A and B

`utils.py`: Common general utils used across the code base


