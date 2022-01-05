# Togo Targeting Replication Code

## Introduction
This repository contains replication code for the paper "Machine Learning and Mobile Phone Data Can Improve Targeting of Humanitarian Assistance" by Emily Aiken, Suzanne Bellue, Dean Karlan, Chris Udry, and Joshua Blumenstock. This readme provides information about the code structure, including where replication code for each figure and table in the paper is located, and information about the data available.

## Code Structure
All code is written in Jupyter Notebooks and raw Python, using Python version 3.6. Replication code for all figures and tables that are generated with code is provided. It is assumed that the reader already has access to featurized mobile phone data matched to survey observations (as shown in the synthetic data provided, see section on "data structure" for more information). The scripts are organized into five notebooks as follows. All notebooks call helper functions from the file `helpers.py`.

### `1survey.ipynb`
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

### `2satellite.ipynb'
Code for aggregation of satellite-based wealth estimates and satellite-based population density estimates for high-resolution poverty mapping. Replication code for the following tables and figures is included in this notebook:
- Figure S2 Panel A
- Figure S8

### `3ml.ipynb`
Code for machine learning from featurized mobile phone data, including matching survey observations to mobile phone records, cross-validation for parameter selection and out-of-sample evaluation, and evaluating feature importances. Replication code for the following tables and figures is included in this notebook:
- Figure S10
- Figure S11
- Table S4
- Table S12
- Table S19

## Data Structure
