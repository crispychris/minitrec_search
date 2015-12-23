Information Retrieval - Project 3
Query Reformulation

Details of software and imports:

This Python script was run on a system using Python 2.7.9
This script also used utf-8 encoding on the text

At the top of the query_processing.py file there are several import statements. Each is used for a pupose:

import os - used for file system manipulation
import time - used for time measurements
import math - used for math operations like log
import sys - used to collect command line arguments
import re - regex for pattern matching in preprocessing
from collections import Counter - counter used to get tf for each term in query
from collections import OrderedDict - an ordered dict came in handy for keeping positional elements in order
import operator - used for sorting tuples when writing results out to a file
from nltk.stem.porter import PorterStemmer - Porter stemmer used for stemming or query terms
import subprocess - used for automated testing purposes
import copy - useful to make deepcopy objects
import csv - writing to csv for testing purposes


Also to be noted are the variable at the top of the file:

CWD = os.getcwd() - gets the current working directory for file paths, all outputs will go here
QUERY_FILE = CWD + "/Mini-Trec-Data/QueryFile/queryfile.txt" - path to queryfile
STOP_WORDS_PATH = CWD + '/Mini-Trec-Data/stops.txt' - path to stop words file
INDICES_PATH = CWD + "/indices/" - appended path to indices folder

below are varaible for file paths for all the inverted indices, lexicons, and doc-term indices
SINGLE_TERM_POSITIONAL_LEXICON = INDICES_PATH + "lexicon_single_term_positional_index_2015-11-03_21-12-56.txt"
SINGLE_TERM_POSITIONAL_II = INDICES_PATH + "single_term_positional_index_2015-11-03_21-12-56.txt"
SINGLE_TERM_LEXICON = INDICES_PATH + "lexicon_single_term_index_2015-11-03_20-26-28.txt"
SINGLE_TERM_II =  INDICES_PATH + "single_term_index_2015-11-03_20-26-28.txt"
STEM_LEXICON = INDICES_PATH + "lexicon_stem_index_2015-11-03_21-47-57.txt"
STEM_II = INDICES_PATH + "stem_index_2015-11-03_21-47-57.txt"
PHRASE_LEXICON = INDICES_PATH + "lexicon_phrase_index_2015-11-03_22-08-18.txt"
PHRASE_II = INDICES_PATH + "phrase_index_2015-11-03_22-08-18.txt"
SINGLE_TERM_DOC_INDEX = INDICES_PATH + "single_term_index_doc_term_index.txt"
SINGLE_TERM_POSITIONAL_DOC_INDEX = INDICES_PATH + "single_term_positional_index_doc_term_index.txt"
STEM_DOC_INDEX = INDICES_PATH + "stem_index_doc_term_index.txt"
PHRASE_DOC_INDEX = INDICES_PATH + "phrase_index_doc_term_index.txt"


BM_25_CONFIG_FILE = CWD + "/bm_25_config.txt" - this is the config file for bm25 tuning parameters
RESULTS_FILE = CWD + "/results.txt" - output file for query results
VS_REFORM_RESULTS_FILE = CWD + "/cosine_reform_results.txt" - output result files for reformulations for each rank method
BM25_REFORM_RESULTS_FILE = CWD  + "/bm25_reform_results.txt"
KL_REFORM_RESULTS_FILE = CWD + "/kl_reform_results.txt"
PROCESSED_QUERIES_FILE = CWD + "/processed_queries.txt" - file outputs the preprocessed queries
EXPANSION_FILE = CWD + "/reduc_expan.txt" - contains config for query reforumlation
PHRASE_THRESHOLD_FILE = CWD + "/threshold_config.txt" - file containing the thresholds for using phrase indices in evaluation, should the number of documents for a query collected from the phrase or positional index surpass the threshold for the respective index, then that index (phrase or positional) will be used. The phrase index threshold and the positional index threshold will by default be set to 20 and 100 respectively. However this file is necessary, otherwise the program will crash

How to run the file:

The program can be run in default mode by typing : python query_expansion.py

The file takes in a maximum of 5 command line arguments. The first setting is the ranking method("cosine_similarity", "bm25", "kl_divergence"). The default is cosine similarity. The second setting is the default index("single_term_index", "stem_index"). This is the index used when phrases are either too infrequent in the documents or are turned off. The default for this setting is single_term_index. The last setting is to set the condition no_phrases. This setting determines whether or nor the phrase index and positional index will be considered when doing retrieval of documents for queries. By default it is set False, which means that phrases are considered (useful for recreating report 2). However, by entering "True" as a command line argument, no_phrases is set to True and only the default index (set by the previos command line arugment is considered in document retrieval).

The fourth setting is either "expansion" or "reduction" (expansion is the default setting)
The fifth setting is "reduc_orig" which if included (default set to false) will not only perform search and rank on reductions but also their original long queries (beware this can take a very long time)

The order of these command line arguments in the command line is irrelevant.

Example commands:

python query_processing.py True
python query_processing.py cosine_similarity stem_index
python query_processing.py reduction 
python query_processing.py single_term_index
python query_processing.py kl_divergence stem_index True

Program Input/Output files:

Input - 
In directory where the Python file exists, the following items must also be present:
Mini-Trec folder from Blackboard
bm_config.txt which contains tuning paramerter values for bm25
threshold_config.txt which contains phrase and positional threshold values
indices folder - this folder containts all the inverted indices, lexicons, and doc-term indices used in query processing (a list of these files may be found at the top of query_processing.py)
reduc_expan.txt - config for reformulation

Note: there is no configuration file for kl-divergence because the tuning parameter was set to average document length
Note: a doc_term index is in the format : doc_id doc_length(sum of each terms tf) list of term_id's in doc

Output:
results.txt - contains the results in the correct format for trec_eval
processed_queries.txt - contains each query's pre-processing results
cosine_reform_results.txt - the output files for the reformulated results for the respective rank method
bm25_reform_results.txt
kl_reform_results.txt


Example stdout:

Chriss-MacBook-Pro-3:InfoRetrieval Chris$ python query_expansion.py expansion
setting up
ranking query results by cosine_similarity
default index: single_term_index
no_phrases set to False
reduction or expansion: expansion
top_n_docs: 5
top_t_terms: 5
query_term_threshold: 5
POSITIONAL_TRESHOLD: 100
PHRASE_THRESHOLD: 20
processing
query number: 372
query 1/21
using defualt index single_term_index with retrieved docs of count 135
original query is: american casino native
reformulated query is: alcohol tribal casino american tribes native

