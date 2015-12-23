Information Retrieval - Project 1
Text Pre-processing and Index Creation

Details of software and imports:

This Python script was run on a system using Python 2.7.9
This script also used utf-8 encoding on the text

At the top of the index_machine.py file there are several import statements. Each is used for a pupose:

import os - used to gain access to the current working directory in order to open and create new files
import re - regex, used for all text pre-processing pattern matching
import time - used to measure time of tasks
import datetime - used to get timestamp allowing unique ouput files to be created
import sys - used to collect command line arguments
from collections import Counter - used to determine document frequency for the terms in a document
from collections import OrderedDict - used to keep the inverted index in order, which is not the default behavior for Python dictionaries
from collections import defaultdict - used the default dictionary for the lexicon as well as as a utility data structure throughout the program
import heapq as h - the priority queue that used in the merging of temp files
from nltk.stem.porter import PorterStemmer - the porter stemmer used in the stem index

Also to be noted are the variables at the top of the file:

DOC_NAMES_FILE = 'doc_names.txt' - this points to a file containing the names of the TREC documents

DOC_FILES_PATH = os.getcwd() + '/Mini-Trec-Data/BigSample/' - this is the file path to the TREC documents
STOP_WORDS_PATH = os.getcwd() + '/Mini-Trec-Data/stops.txt' - this is the file path to the stop words
MERGE_FILE_SIZE = 25 - this number is how many temp files are merged at once. If more than 25 temp files exist the program will merge 25 at a time into merge files until there are less than 25 merge files then write the inverted index

How to run the file:

The file can be run in default mode by using "python index_machine.py"

The file takes in two command line arguments: the index type (single_term_index, single_term_positional_index, stem_index, or phrase_index) and the memory requirement (1000, 10000, 100000, unlimited). These command line arguments do not need to be entered in any particular order. The default arguments are singler_term_index and unlimited memory

Example command:

python index_machine.py stem_index 10000
python index_machine.py phrase_index unlimited
python index_machine.py 1000 - here the default index type will be single_term_index
python index_machine.py single_term_index - the default memory requirement will be unlimited

The program outputs both to the stdout and to the directory from where it was called

The program will output which large document file it is processing, if it made a temp file, as well as any preliminary merges it makes on temp files before merging them into the index. It will also output the time it took to make the temp files, the time it took to merge the temp files, and the total time elapsed.

Example run stdout:

Chriss-MacBook-Pro-2:InfoRetrieval Chris$ python index_machine.py 100000
running index_machine.py with command line inputs: -, 100000
default index_type is single_term_index and default memory_restriction is unlimited
processing document: fr940104.0
processing document: fr940104.2
processing document: fr940128.2
processing document: fr940303.1
making temp0
processing document: fr940405.1
processing document: fr940525.0
processing document: fr940617.2
processing document: fr940810.0
making temp1
processing document: fr940810.2
making temp2
processing document: fr941006.1
processing document: fr941206.1
making temp3
just made temp files, will merge now
459.217666864
building inverted index from temp files
merged temp file
2.96135115623
total time elapsed
462.631798029

The program also outputs several things into the directory from which it was called. 

If the program needs to make temp files it will create a directory in the form of temp2015-10-15_19/49/37. In this directory the program will store all temp and merge files. Finally, the program will create an inverted index file in the format of single_term_positional_index_2015-10-15_19/49/37.txt



