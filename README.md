# Classifying 17th century publications based on their titles

## Summary
This repository contains the notebooks used to classify 17th century publications based on their titles. We use two approaches.

The first model is a biterm topic model. The second model using hdbscan to classify the embedded titles with added covariates.

## Requirements
* Python 3.1+ and ability to install libraries.
* Ability to view and execute Jupyter notebooks (https://jupyter.org).
* ESTC input data.

## Files:
Files are designed to be executed in numerical order for each approach. Regardless of the approach, run 00-estc_btm_prep.ipynb first.

#### `00-estc_btm_prep.ipynb`
This file takes the raw ESTC input data and produces a processed file ready for further analysis. Specfically the script:
1) Removes stopwords (default ntlk and some additional from analysis)
2) Keeps only publications labelled as 'English'
3) Filters file to only contain publications from 17th century
4) Removes exact and near duplicates from the same publication year.
5) Stems words using ntlk PorterStemmer
6) After all that, removes titles with fewer than 4 words remaining.

Now you can run either model in the follow order:

### Biterm Topic Model approach

#### `btm_approach/01-search_btm_wide.ipynb`
This file takes the output from _00-estc_btm_prep.ipynb_ and runs a series of Biterm Topic Models across a wide range of topic numbers. It evaluates those models using semantic coherance and entropy.

#### `btm_approach/02-search_btm_narrow.ipynb`
For the researcher, based on the results from 01-search_btm_wide.ipynb, to narrow their search for the optimal number of topics.

#### `btm_approach/02a-search_btm_try_options.ipynb`
An optional script for researchers to 'try out' full models that were identified in 02-search_btm_narrow.ipynb.

#### `btm_approach/03-run_btm.ipynb`
Run the final Biterm Topic Model once the optimal specification has been chosen. Results/outputs are saved.

#### `btm_approach/04-create_results.ipynb`
This file creates some more user-friendly output files for users to explore the results. In particular, joins back onto the original input data.

### Classification including covariates

#### `01-further_data_prep.ipynb`
Further dara prep before the second classification. Includes embedding at several different vector dimensions. Note, the choice of model for embedding is trading off performance and speed. Could be changed in the future.

#### `02-clustering.ipynb`
Classification using hdbscan.
