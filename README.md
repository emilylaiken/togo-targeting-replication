# Togo Targeting Replication Code

## Introduction
This repository contains replication code for the paper "Machine Learning and Mobile Phone Data Can Improve Targeting of Humanitarian Assistance" by Emily Aiken, Suzanne Bellue, Dean Karlan, Chris Udry, and Joshua Blumenstock. This readme provides information about the code structure, including where replication code for each figure and table in the paper is located, and information about the data available.

## Code Structure
All code is written in Jupyter Notebooks and raw Python, using Python version 3.6. Replication code for all figures and tables that are generated with code is provided. It is assumed that the reader already has access to featurized mobile phone data matched to survey observations (as shown in the synthetic data provided, see section on "data structure" for more information). The scripts are organized into five notebooks as follows. All notebooks call helper functions from the file `helpers.py`.

#### `1survey.ipynb`
Code for all figures and tables that are generated from mainly from survey data, including the calculation of the PMT and asset index, statistics on phone ownership, summary statistics from surveys, and information on weighting and response weights. Replication code for the following tables and figures is included in this notebook:
- Figure S1
- Figure S3
- Figure S6
- Figure S7
- Figure S14
- Figure S15
- Table S7
- Table S8
- Table S9
- Table S10
- Table S11
- Table S16

#### `2satellite.ipynb`
Code for aggregation of satellite-based wealth estimates and satellite-based population density estimates for high-resolution poverty mapping. Replication code for the following tables and figures is included in this notebook:
- Figure S2 Panel A
- Figure S8

#### `3ml.ipynb`
Code for machine learning from featurized mobile phone data, including matching survey observations to mobile phone records, cross-validation for parameter selection and out-of-sample evaluation, and evaluating feature importances. Replication code for the following tables and figures is included in this notebook:
- Figure S10
- Figure S11
- Table S4
- Table S12
- Table S19

#### `4targeting.ipynb` 
Code for targeting simulations, including producing targeting tables of all kinds, ROC curves and precision vs. recall curves, and analysis of social welfare. Replication code for the following tables and figures is included in this notebook:
- Figure 1
- Figure 2
- Figure S2 Panel B
- Figure S12
- Figure S13
- Table 1
- Table S1
- Table S2
- Table S3
- Table S5
- Table S13
- Table S14
- Table S15

#### `5fairness.ipynb`
Code for fairness audits of targeting algorithms across potentially sensitive characteristics. Replication code for the following tables and figures is included in this notebook:
- Figure 3
- Figure S4
- Figure S5

## Data Structure
Data files that are publically available are included in this repo; for datasets that are not publically available we have included synthetic (randomly generated) data in the same format and with the same schema. Synthetic data are produced with the notebook `data/generate_synthetic_data.ipynb'. The data files are as follows:
- `data/survey2018.csv` and `data/survey2020.csv`: Synthetic data with the schema of the 2018 and 2020 survey datasets, respectively. The 2018 survey is a household survey dataset; the 2020 survey is at the individual level.
- `survey_indiv2018.csv`: Synthetic individual-level survey data associated with the households from the 2018 survey dataset; used only to calculate statistics on individual-level phone ownership.
- `data/features2018.csv` and `data/features2020.csv`: Synthetic data representing a set of featurized mobile phone data. In this file features are randomly generated; in reality they are calculated from raw mobile phone records using open source library bandicoot. Features are provided for a subset of the phone numbers (observations) in each of the 2018 and 2020 synthetic survey datasets.
- `data/single_feature2018.csv`: Synthetic data representing the "single mobile phone feature" used as a counterfactual targeting method in the paper. The single mobile phone feature is provided for the same set of phone numbers that are associated with full mobile phone featurization.
- `data/inferred_home_location2018.csv`: Synthetic data representing inferred home locations from mobile phone data. Here home locations are chosen at random; in reality they are inferred from raw mobile phone metadata. Home locations are provided for the same subset of survey observations that are associated with full mobile phone features; they are provided at the prefecture and canton level.
- `data/shapefiles`: Shapefiles used for poverty mapping. Shapefiles are publicly available from the Humantiarian Data Exchange (https://data.humdata.org/dataset/togo-cod-ab). 
- - `data/shapefiles/regions.geojson`: Shapefile for regions (admin-1 level)
- - `data/shapefiles/prefectures.geojson`: Shapefile for prefectures (admin-2 level)
- - `data/shapefiles/cantons.geojson`:  Shapefile for cantons (admin-3 level)
- `data/satellite`: Data for satellite-based poverty mapping. 
- - `data/satellite/wealth.csv`: Tile-level relative wealth estimates from satellite imagery publicly available on the Humanitarian Data Exchange (https://data.humdata.org/dataset/relative-wealth-index)
- - `data/satellite/pop.csv`: Tile-level population density estimates from satellite imagery publicly available on the Humanitarian Data Exchange (https://data.humdata.org/dataset/highresolutionpopulationdensitymaps-tgo)

## Running the Code
After installing the necessary packages, the code will run out of the box (using the data files located in the `data` subfolder). Outputs will be written to the `outputs` subfolder, divided into output folders for `ml`, `survey`, and `targeting`. Many of the figures and tables are not written to files but rather displayed only in the Jupyter notebooks.
