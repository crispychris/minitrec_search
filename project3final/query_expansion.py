# coding= utf-8
import os
import datetime
import time
import math
import sys
import re
from collections import Counter
from collections import OrderedDict
import operator
from nltk.stem.porter import PorterStemmer
import subprocess
import copy
import csv

CWD = os.getcwd()
QUERY_FILE = CWD + "/Mini-Trec-Data/QueryFile/queryfile.txt"
STOP_WORDS_PATH = CWD + '/Mini-Trec-Data/stops.txt'
INDICES_PATH = CWD + "/indices/"
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
BM_25_CONFIG_FILE = CWD + "/bm_25_config.txt"
RESULTS_FILE = CWD + "/results.txt"
VS_REFORM_RESULTS_FILE = CWD + "/cosine_reform_results.txt"
BM25_REFORM_RESULTS_FILE = CWD  + "/bm25_reform_results.txt"
KL_REFORM_RESULTS_FILE = CWD + "/kl_reform_results.txt"
PROCESSED_QUERIES_FILE = CWD + "/processed_queries.txt"
PHRASE_THRESHOLD_FILE = CWD + "/threshold_config.txt"
EXPANSION_FILE = CWD + "/reduc_expan.txt"
TREC_EVAL_FILE = CWD + "/trec_eval"


time_seconds = time.time()
time_stamp = datetime.datetime.fromtimestamp(time_seconds).strftime('%Y-%m-%d_%H-%M-%S')


class Query:

	def __init__(self, number, title, narrative):
		self.id = number
		self.query_text = title
		self.narrative = narrative

	def print_query(self):
		print (self.id, self.query_text, self.narrative)

	def __str__(self):
		return self.id +"\n" +  self.query_text +"\n" + self.narrative

class Index:

	def __init__(self, lexicon, inverted_index, doc_term_id_index):
		self.lexicon = lexicon
		self.inverted_index = inverted_index
		self.doc_term_id_index = doc_term_id_index

class Query_Processor:

	def __init__(self, rank_method, default_index, no_phrases, r_e, test, reduc_orig, query_list, **indices):
		self.rank_method = rank_method
		self.default_index = default_index
		self.query_list = query_list
		self.reduc_expan = r_e
		self.no_phrases = no_phrases
		self.reduc_orig = reduc_orig
		self.query_results = []
		self.results = []
		self.reformulated_results = []
		self.indices = indices
		self.full_normalized_query = {}
		self.reformulated_results_pairs = []
		self.total_documents = 1765
		self.dates = []
		self.test = test
		self.K1 = 0
		self.K2 = 0
		self.B = 0
		self.POSITIONAL_TRESHOLD = 100
		self.PHRASE_THRESHOLD = 20
		self.c_size = {"single_term_index":None, "single_term_positional_index":None, "stem_index":None, "phrase_index":None}

		bm_25_config = open(BM_25_CONFIG_FILE, 'r')
		for line in bm_25_config:
			if "#" not in line:
				splits = line.split(" ")
				self.K1 = float(splits[0])
				self.K2 = int(splits[1])
				self.B = float(splits[2])

		threshold_config = open(PHRASE_THRESHOLD_FILE, 'r')
		for row in threshold_config:
			if "#" not in row:
				splits = row.split(" ")
				self.POSITIONAL_TRESHOLD = int(splits[0])
				self.PHRASE_THRESHOLD = int(splits[1])

		reduc_expan_config = open(EXPANSION_FILE, 'r')
		for row in reduc_expan_config:
			if "top_n_docs" in row:
				self.top_n_docs = int(row.split("\t")[1].strip("\n"))
				print "top_n_docs: " + str(self.top_n_docs)
			if "top_t_terms" in row:
				self.top_t_terms = int(row.split("\t")[1].strip("\n"))
				print "top_t_terms: " + str(self.top_t_terms)

			if "query_term_threshold" in row:
				self.query_term_threshold = int(row.split("\t")[1].strip("\n"))
				print "query_term_threshold: " + str(self.query_term_threshold)

		print "POSITIONAL_TRESHOLD: " + str(self.POSITIONAL_TRESHOLD)
		print "PHRASE_THRESHOLD: " + str(self.PHRASE_THRESHOLD)

		self.date_dict = {"january":1, "jan":1, "february":2, "feb":2, "march":3, "mar":3, "april":4, "apr":4, "may":5, "jun":6, "june":6, "july":7, "jul":7, "aug":8, "august":8, "sept":9, "september":9, "oct":10, "october":10, "nov":11, "november":11, "dec":12, "december":12}
		self.common_prefixes = ["un", "re", "in", "im", "ir", "ill", "dis", "en", "em", "non", "in", "im", "over", "mis", "sub", "pre", "inter", "fore", "de", "trans", "super", "semi", "anti", "mid", "under"]
		self.forbidden_char = ['-', '/', ',', '[', ']', '(', ')', '@', '^', '#', '!', '%', '&', '*', '?', '|', ';', '{', '}', ':', '`', '\'', '_', '=', '+', '<', '>', '\\', '.', '~']
		self.stop_words = []
		stop_words_file = open(STOP_WORDS_PATH, 'r')
		for stop in stop_words_file:
			self.stop_words.append(stop.strip('\n'))

	def stringify_date(self, month_int, month_string, day, year):

		#month is string or int
		#day and year are int

		month_valid = False
		day_valid = False
		month_val = 0
		day_val = 0
		year_val = 0

		int_day= int(day)
		int_year = int(year)

		normalized = ""

		#deal with jan. case
		if month_string != None:
			month_string = month_string.replace(".", "")

		if month_string in self.date_dict.keys():

				month_val = str(self.date_dict[month_string])
				month_valid = True

		elif month_int != None and month_int >=1 and month_int <=12:

				month_val = str(month_int)
				month_valid = True


		if month_val != "10" and month_val != "11" and month_val != "12":
					
			month_val = "0"+month_val

		if int_day <=31 and int_day >=1:

			day_val = str(day)
			day_valid = True
						
			if day_val == "1" or day_val == "2" or day_val == "3" or day_val == "4" or day_val == "5" or day_val == "6" or day_val == "7" or day_val == "8" or day_val == "9":
							
				day_val = "0"+day_val

		if month_valid == True and day_valid == True:

			if int_year >= 1900 and int_year <= 2016:

				year_val = str(year)
				normalized = month_val + '/' + day_val + '/' + year_val

			elif int_year >=1 and int_year <=9:

				year_val = "200"+str(year)
				normalized = month_val + '/' + day_val + '/' + year_val

			elif int_year >=10 and int_year <=16:

				year_val = "20" + str(year)

				normalized = month_val + '/' + day_val + '/' + year_val

			elif int_year >=17 and int_year <=99:

				year_val = "19"+str(year)

				normalized = month_val + '/' + day_val + '/' + year_val

		return normalized

	def normalize(self, query):

		#make all lowercase
		query = query.lower()
		
		#redo unicode, just in case
		query = query.replace("&racute;", u"ŕ")
		query = query.replace("&atilde;", u"ã")
		query = query.replace("&hyph;", u"-")
		query = query.replace("&eacute", u"é")
		query = query.replace("&ntilde;", u"ñ")
		query = query.replace("&agrave;", u"à")
		query = query.replace("&oacute;", u"ó")
		query = query.replace("&egrave;", u"è")
		query = query.replace("&ocirc;", u"ô")
		query = query.replace("&aacute;", u"á")
		query = query.replace("&uuml;", u"ü")
		query = query.replace("&rsquo;", u"'")
		query = query.replace("&iacute;", u"í")
		query = query.replace("&cent;", u"¢")
		query = query.replace("&ccedil;", u"ç")

		#remove other ampersand strings
		ampersand_char = re.findall("&\w+;", query)
		ampersand_set = set(ampersand_char)

		for item in ampersand_set:
			query = query.replace(item, " ")

		#remove characters that have no use, even in speical tokens
		query = query.replace("<", " ")
		query = query.replace(">", " ")
		query = query.replace("\n", " ")


		#normalizes the dates for a few date formats
		dates = []
	
		#find all dates in doc_text
		mmddyyyy_yy= re.findall("(\d+[-\\/]\d+[-\\/]\d+)\D", query)
		mmmddyyyy = re.findall("((?:jan|feb|mar|apr|jun|jul|aug|sept|oct|nov|dec)[,\\.\s-]+\d+[\s,-]+\d+)\D",query)
		nameddyyyy = re.findall("((?:january|february|march|april|may|june|july|august|september|october|november|december)\s+\d+(?:,|\s)+\d+)\D", query)
		ddnameyyyy = re.findall("\D(\d{1,2}[-\s]+(?:january|february|march|april|may|june|july|august|september|october|november|december|jan|feb|mar|apr|jun|jul|aug|sept|oct|nov|dec)[-\\.,\s]+\d+)[^0-9]", query)

		#deal with mmddyyyy_yy

		"""a few cases:

			mm-dd-yyyy
			mm-dd-yy
			dd-mm-yyyy
			dd-mm-yy

		"""

		converted_mmddyyyy_yy = {}

		for original in mmddyyyy_yy:

			done = False

			split = original.split("-")
			if split[0] == original:
				split = original.split("/")

			if len(split) >= 3:#gets rid of some non-dates

				val1 = int(split[0])
				val2 = int(split[1])
				val3 = int(split[2])

				#case: mm-dd-yyyy 
				#this case is the most standard so unless the mm value is greater than 12 I will take it to be the month
				if val1 >=1 and val1 <= 12:
					if val2 >= 1 and val2 <= 31:
						if val3 >= 1900 and val3 <= 2016:

							normalized = self.stringify_date(val1, None, val2, val3)

							converted_mmddyyyy_yy.update({original:normalized})
							done = True

				#case: mm-dd-yy
				if done == False:
					if val1 >=1 and val1 <= 12:
						if val2 >= 1 and val2 <= 31:
							if val3 >= 1 and val3 <= 99:

								normalized = self.stringify_date(val1, None, val2, val3)

								converted_mmddyyyy_yy.update({original:normalized})
								done = True

				#case dd-mm-yyyy
				if done == False:
					if val1 >=1 and val1 <= 31:
						if val2 >= 1 and val2 <= 12:
							if val3 >= 1900 and val3 <= 2016:

								normalized = self.stringify_date(val2, None, val1, val3)

								converted_mmddyyyy_yy.update({original:normalized})
								done = True

				#case dd-mm-yy
				if done == False:
					if val1 >=1 and val1 <= 31:
						if val2 >= 1 and val2 <= 12:
							if val3 >= 1 and val3 <= 99:

								normalized = self.stringify_date(val2, None, val1, val3)

								converted_mmddyyyy_yy.update({original:normalized})

		if converted_mmddyyyy_yy != {}:

			for key in converted_mmddyyyy_yy.keys():

				query = query.replace(key, converted_mmddyyyy_yy[key])
				dates.append(converted_mmddyyyy_yy[key])

		#deal with ddnameyyyy
		
		converted_ddnameyyyy = {}

		for original in ddnameyyyy:

			month = ""
			day =0
			year=0

			splits = original.split(" ")
			splits = [x for x in splits if x != ""]#get rid of blanks

			month = splits[1]
			day = int(splits[0])
			year = int(splits[2])

			normalized = self.stringify_date(None, month, day, year)
							
			converted_ddnameyyyy.update({original:normalized})

		if converted_ddnameyyyy != {}:

			for key in converted_ddnameyyyy.keys():

				query = query.replace(key, converted_ddnameyyyy[key])
				dates.append(converted_ddnameyyyy[key])

		#deal with mmmddyyyy

		converted_mmmddyyyy = {}

		for original in mmmddyyyy:

			month = ""
			day = 0
			year = 0

			item = original.replace(",", " ")#remove commas
			item = item.replace(".", " ")#remove commas
			splits = item.split(" ")
			splits = [x for x in splits if x != ""]#get rid of blanks

			if len(splits) >= 3:
				
				month = splits[0]
				day = int(splits[1])
				year = int(splits[2])

				normalized = self.stringify_date(None, month, day, year)

				converted_mmmddyyyy.update({original:normalized})
				
		if converted_mmmddyyyy != {}:
			for key in converted_mmmddyyyy.keys():

				query = query.replace(key, converted_mmmddyyyy[key])
				dates.append(converted_mmmddyyyy[key])

		#deal with nameddyyyy case
		
		converted_nameddyyyy = {}

		for original in nameddyyyy:

			month = ""
			day = 0
			year = 0

			item = original.replace(",", " ")#remove commas
			splits = item.split(" ")
			splits = [x for x in splits if x != ""]#get rid of blanks

			if len(splits) >= 3:#filters out the weird dates

				month = splits[0]
				day = splits[1]
				year = splits[2]

				normalized = self.stringify_date(None, month, day, year)

				converted_nameddyyyy.update({original:normalized})

		if converted_nameddyyyy != {}:

			for key in converted_nameddyyyy.keys():

				query = query.replace(key, converted_nameddyyyy[key])
				dates.append(converted_nameddyyyy[key])

		self.dates = dates
		return query

	def split_on_period(self, word_list):

		new_word_list = []

		for item in word_list:

			result = re.sub(r"\.(?!\d)", " ", item)
			result_list = result.split(" ")
			result_list = [x for x in result_list if x != ""]

			for it in result_list:

				new_word_list.append(it)


		return new_word_list

	def split_on_char(self, charac, word_list):

		new_word_list = []


		for item in word_list:

			result = item.replace(charac, " ")
			result_list = result.split(" ")
			result_list = [x for x in result_list if x != ""]

			for it in result_list:

				new_word_list.append(it)

		return new_word_list

	def normalize_phrase(self, query):

		#make all lowercase
		query = query.lower()

		#redo unicode
		query = query.replace("&racute;", u"ŕ")
		query = query.replace("&atilde;", u"ã")
		query = query.replace("&hyph;", u"-")
		query = query.replace("&eacute", u"é")
		query = query.replace("&ntilde;", u"ñ")
		query = query.replace("&agrave;", u"à")
		query = query.replace("&oacute;", u"ó")
		query = query.replace("&egrave;", u"è")
		query = query.replace("&ocirc;", u"ô")
		query = query.replace("&aacute;", u"á")
		query = query.replace("&uuml;", u"ü")
		query = query.replace("&rsquo;", u"'")
		query = query.replace("&iacute;", u"í")
		query = query.replace("&cent;", u"¢")
		query = query.replace("&ccedil;", u"ç")

		#remove other ampersand strings
		ampersand_char = re.findall("&\w+;", query)
		ampersand_set = set(ampersand_char)

		for item in ampersand_set:
			query = query.replace(item, " ")
		
		# #check IP address
		ip_address = re.findall(r"\b(?:\d{1,3}\.){3}\d{1,3}\s", query)

		if ip_address != []:
			
			for item in ip_address:

				query = query.replace(item, "<")


		# #check URL

		urls = re.findall(r'(http|ftp|https)://([\w_-]+(?:(?:\.[\w_-]+)+))([\w.,@?^=%&:/~+#-]*[\w@?^=%&/~+#-])?', query)

		if urls != []:

			for item in urls:
				query = query.replace(item, "<")


		# #check for email

		emails = re.findall(r'\b[a-z0-9\._%+-]+@[a-z0-9.-]+\.[a-z]{2,4}\b', query)
		
		if emails != []:

			for item in emails:
				query = query.replace(item, "<")

		# #standardize digit format

		numerals = re.findall(r'((?:\d+[,\.])+\d+)\b', query)
		#print numerals

		if numerals != []:

			for item in numerals:

					query = query.replace(item, "<")


		# #check for monetary symbols with numbers

		money = re.findall(r'(?:[$€¢£](?:\d+[,\.])+\d+)', query)

		if money != []:

			for cash in money:

				query = query.replace(cash, "<")

		# #deal with abbreviations

		abbs = re.findall(r"(?:[a-z]\.){2,100}", query)
		if abbs != []:

			for thing in abbs:
				query = query.replace(thing, "<")

		#remove hyphenations

		hyphs = re.findall(r"\b([\d\w]+-[\d\w]+)\b", query)

		if hyphs != []:

			for il in hyphs:
				query = query.replace(il, "<")

		#remove number

		nums = re.findall(r"\d+", query)

		if nums != []:

			for yu in nums:
				query = query.replace(yu, "<")

		# #find all dates in doc_text
		mmddyyyy_yy= re.findall("(\d+[-\\/]\d+[-\\/]\d+)\D", query)

		for d in mmddyyyy_yy:

			query = query.replace(d, "<")

		mmmddyyyy = re.findall("((?:jan|feb|mar|apr|jun|jul|aug|sept|oct|nov|dec)[,\\.\s-]+\d+[\s,-]+\d+)\D", query)

		for d in mmmddyyyy:

			query = query.replace(d, "<")

		nameddyyyy = re.findall("((?:january|february|march|april|may|june|july|august|september|october|november|december)\s+\d+(?:,|\s)+\d+)\D", query)

		for d in nameddyyyy:

			query = query.replace(d, "<")

		ddnameyyyy = re.findall("\D(\d{1,2}[-\s]+(?:january|february|march|april|may|june|july|august|september|october|november|december|jan|feb|mar|apr|jun|jul|aug|sept|oct|nov|dec)[-\\.,\s]+\d+)[^0-9]", query)

		for d in ddnameyyyy:

			query = query.replace(d, "<")
		
		return query

	def check_for_stop(self, word):

		if word in self.stop_words:
			return "<"
		else:
			return word

	def split_on_stop(self, phrase_list):

		phrase_list = phrase_list.split(" ")
		phrase_list = [th for th in phrase_list if th != ""]
		phrase_list = map(self.check_for_stop, phrase_list)


		if len(phrase_list) >1:
			return  phrase_list
		else:
			return []

	def remove_nums(self, text):

		text = text.replace("1", "<")
		text = text.replace("2", "<")
		text = text.replace("3", "<")
		text = text.replace("4", "<")
		text = text.replace("5", "<")
		text = text.replace("6", "<")
		text = text.replace("7", "<")
		text = text.replace("8", "<")
		text = text.replace("9", "<")
		text = text.replace("0", "<")

		return text

	#makes two and three words phrases out of a group of more than two or three words
	def phrazify(self, phrase_list):

		length = len(phrase_list)
		return_list = []

		idx = 0
		phrase_length = 3

		while idx +phrase_length -1 < length:

			return_list.append(" ".join(phrase_list[idx:idx +phrase_length]))
			idx += 1


		idx = 0
		phrase_length = 2

		while idx + phrase_length -1 < length:

			return_list.append(" ".join(phrase_list[idx:idx + phrase_length]))
			idx += 1

		return return_list

	def find_phrases(self, text):

		phrases = []

		splits = text.split("<")
		for splitter in self.forbidden_char:

			splits = [x.split(splitter) for x in splits]
			splits = [item for sublist in splits for item in sublist]

		new_splits = map(self.split_on_stop, splits)

		new_splits = [ite for ite in new_splits if ite != []]
		
		new_splits = map(lambda l: " ".join(l), new_splits)

		new_splits = map(self.remove_nums, new_splits)

		new_splits = map(lambda l:l.split("<"), new_splits)

		new_splits = [item for sublist in new_splits for item in sublist]

		brand_new = []
		for ns in new_splits:

			ns_split = ns.split(" ")
			ns_split = [nss for nss in ns_split if nss != ""]
			if len(ns_split) > 1:
				brand_new.append(ns)

		brand_new = map(lambda l:l.rstrip(" ").lstrip(" "), brand_new)

		for iti in brand_new:

			phrase_split = iti.split(" ")

			if len(phrase_split) == 2:

				tmp = []
				tmp.append(" ".join(phrase_split))

				phrases.append(tmp)

			elif len(phrase_split) == 3:

				tmp = []
				tmp.append(" ".join(phrase_split[0:2]))
				tmp.append(" ".join(phrase_split[1:3]))
				tmp.append(" ".join(phrase_split[0:3]))

				phrases.append(tmp)

			elif len(phrase_split) >= 4:

				phrases.append(self.phrazify(phrase_split))

		phrases = [item for sublist in phrases for item in sublist]

		return phrases

	def tokenize_single_stem(self, word_input):

		dates = self.dates
		word = word_input
		first = word

		#depends on the index type

		#single term index

		tokens = []
		
		#check IP address
		ip_address = re.findall(r"\b(?:\d{1,3}\.){3}\d{1,3}\s", word)

		if ip_address != []:
			
			for item in ip_address:

				tokens.append(item)
				word = word.replace(item, " ")


		#check URL

		urls = re.findall(r'(http|ftp|https)://([\w_-]+(?:(?:\.[\w_-]+)+))([\w.,@?^=%&:/~+#-]*[\w@?^=%&/~+#-])?', word)

		if urls != []:

			for item in urls:
				tokens.append(item)
				word = word.replace(item, " ")


		#check for email

		emails = re.findall(r'\b[a-z0-9\._%+-]+@[a-z0-9.-]+\.[a-z]{2,4}\b', word)
		
		if emails != []:

			for item in emails:
				tokens.append(item)
				word = word.replace(item, " ")

		#standardize digit format

		numerals = re.findall(r'((?:\d+[,\.])+\d+)\b', word)
		#print numerals

		if numerals != []:

			for item in numerals:

				if len(item.split(".")) <= 2:#nothing with two decimal points, not a number

					new_numeral = ""

					number = item.replace(",", "")
					int_number = int(number.split(".")[0])
					float_number = float(number)
					#we need to chop off the trailling 0 decimals

					diff = float_number - int_number

					if diff > 0:#there is a decimal, use the float version

						new_numeral = str(float_number)

					else:#we chop off the decimals by using the int version 

						new_numeral = str(int_number)

					word = word.replace(item, new_numeral)


		#check for monetary symbols with numbers

		money_vals = []

		money = re.findall(r'(?:[$€¢£](?:\d+[,\.])+\d+)', word)

		if money != []:

			for cash in money:

				to_token = ""

				new_cash = cash.replace(",", "")
				money_sign = new_cash[0]
				money_value = new_cash[1:len(new_cash)]
				int_cash = int(money_value.split(".")[0])
				float_cash = float(money_value)
				#we need to chop off the trailling 0 decimals

				diff_cash = float_cash - int_cash

				if diff_cash > 0:#there is a decimal, use the float version

					to_token =  money_sign + str(float_cash)

				else:#we chop off the decimals by using the int version 

					to_token = money_sign + str(int_cash)

				word = word.replace(cash, to_token)
				money_vals.append(word)



		#deal with abbreviations

		abbs = re.findall("(?:[a-z]\.){2,100}", word)
		if abbs != []:

			for thing in abbs:
				word = word.replace(thing, thing.replace(".", ""))

		#now can remove periods and commas and the lot

		#moving into the final phase where we get all the tokens

		tok = []
		tok.append(word)

		new_word_list = []

		no_dates = True

		if dates != []:
			for date in dates:
				if date in word:
					no_dates = False


		if no_dates == True:
			tok = self.split_on_char("/", tok)

		#special way for periods, dont want to remove periods relating to numbers
		tok = self.split_on_period(tok)


		tok = self.split_on_char(",", tok)
		tok = self.split_on_char("[", tok)
		tok = self.split_on_char("]", tok)
		tok = self.split_on_char("(", tok)
		tok = self.split_on_char(")", tok)
		tok = self.split_on_char("@", tok)
		tok = self.split_on_char("^", tok)
		tok = self.split_on_char("#", tok)
		tok = self.split_on_char("!", tok)
		tok = self.split_on_char("%", tok)
		tok = self.split_on_char("&", tok)
		tok = self.split_on_char("*", tok)
		tok = self.split_on_char("?", tok)
		tok = self.split_on_char("|", tok)
		tok = self.split_on_char(";", tok)
		tok = self.split_on_char("{", tok)
		tok = self.split_on_char(":", tok)
		tok = self.split_on_char("}", tok)
		tok = self.split_on_char("``", tok)
		tok = self.split_on_char("'", tok)
		tok = self.split_on_char("`", tok)
		tok = self.split_on_char("_", tok)
		tok = self.split_on_char("=", tok)
		tok = self.split_on_char("+", tok)
		tok = self.split_on_char("\"", tok)

		#dealing with hyphens

		#three cases
		#alpha-digit
		#digit-alpha
		#alpha-alpha

		for thing in tok:

			if "-" in thing:

				alpha_digit_small = re.findall(r"[^\d]{1,2}[-]\d+", thing)

				if alpha_digit_small != []:

					for entity1 in alpha_digit_small:

						new_word_list.append(entity1.split("-")[0]+entity1.split("-")[1])

				alpha_digit_large = re.findall(r"[^\d]{3,}[-]\d+", thing)

				if alpha_digit_large != []:

					for entity2 in alpha_digit_large:

						new_word_list.append(entity2.split("-")[0]+entity2.split("-")[1])
						new_word_list.append(entity2.split("-")[0])

				digit_alpha_small = re.findall(r"\d+[-][^\d]{1,2}", thing)

				if digit_alpha_small != []:

					for entity3 in digit_alpha_small:
						new_word_list.append(entity3.split("-")[0]+entity3.split("-")[1])

				digit_alpha_large = re.findall(r"(?:\d+-[^\d]{3,})", thing)

				if digit_alpha_large != []:

					for entity4 in digit_alpha_large:

						new_word_list.append(entity4.split("-")[0]+entity4.split("-")[1])
						new_word_list.append(entity4.split("-")[1])

				alpha_alpha = re.findall(r"[^\d]+(?:[-][^\d]+)+", thing)

				if alpha_alpha != []:

					for entity5 in alpha_alpha:

						if entity5.split("-")[0] in self.common_prefixes:

							new_word_list.append(entity5.split("-")[0]+entity5.split("-")[1])
							new_word_list.append(entity5.split("-")[1])

						else:

							w = entity5.split("-")
							new_word_list.append("".join(w))

							for term in w:

								new_word_list.append(term)

				digit_digit = re.findall(r"\d+(?:[-][\d]+)+", thing)

				if digit_digit != []:

					for entity6 in digit_digit:

						nd = entity6.split("-")
						new_word_list.append("".join(nd))

			else:

				new_word_list.append(thing)

		new_word_list = [x for x in new_word_list if x != ""]	

		return tokens + new_word_list


	def index_query_terms(self):

		index_types = ["single_term_index", "single_term_positional_index", "stem_index", "phrase_index",
		"single_term_index_narrative", "single_term_positional_index_narrative",
		 "stem_index_narrative", "phrase_index_narrative"]

		for query_id in self.full_normalized_query.keys():
			for index in index_types:

				if index != "single_term_positional_index" and index != "single_term_positional_index_narrative":
					self.full_normalized_query[query_id][index] = Counter(self.full_normalized_query[query_id][index])
				else:
					counted = Counter(self.full_normalized_query[query_id][index])
					ordered = OrderedDict()
					for term in self.full_normalized_query[query_id][index]:
						ordered.update({term:counted[term]})
					self.full_normalized_query[query_id][index] = ordered


	def print_query_list(self,qlist):
		for item in qlist:
			item.print_query()	

	def write_queries(self):

		fil = open(PROCESSED_QUERIES_FILE, 'w')

		for item in self.full_normalized_query.keys():
			fil.write("#" + item + "\n")
			for thing in self.full_normalized_query[item].keys():
				fil.write(thing + "\t")
				for obj in self.full_normalized_query[item][thing]:
					fil.write(obj + ",")
				fil.write("\n")

		fil.close()

	def process(self):

		#reuse a lot of code from project 1
		qlist = self.query_list
		full_normalized_query = self.full_normalized_query
		single_term_tokenization = []
		single_term_tokenization_narrative = []
		single_term_positional_tokenization = []
		single_term_positional_tokenization_narrative = []
		phrase_tokenization = []
		phrase_tokenization_narrative = []

		stemmer = PorterStemmer()

		#fill up the query dictionary
		for q in qlist:
			full_normalized_query.update({q.id:{"single_term_index":[], "single_term_positional_index":[], "stem_index":[], "phrase_index":[], "single_term_index_narrative":[], "single_term_positional_index_narrative":[], "stem_index_narrative":[], "phrase_index_narrative":[]}})

		#normalize quries for all common rules
		date_normalized = [(q.id, self.normalize(q.query_text)) for q in qlist]
		date_normalized_narrative = [(q.id, self.normalize(q.narrative)) for q in qlist]
		phrase_normalized = [(z[0], self.normalize_phrase(z[1])) for z in date_normalized]
		phrase_normalized_narrative = [(z[0], self.normalize_phrase(z[1])) for z in date_normalized_narrative]

		#create single term positional tokenization aka dont remove stop words
		for item in date_normalized:

			words = item[1].split(" ")
			#remove blanks
			words = [x for x in words if x != ""]
			tokens = map(self.tokenize_single_stem, words)
			tokens = [thing for sublist in tokens for thing in sublist]

			single_term_positional_tokenization.append((item[0], tokens))
			full_normalized_query[item[0]]["single_term_positional_index"] = tokens


		for selection in date_normalized_narrative:

			words = selection[1].split(" ")
			#remove blanks
			words = [x for x in words if x != ""]
			tokens = map(self.tokenize_single_stem, words)
			tokens = [thing for sublist in tokens for thing in sublist]

			single_term_positional_tokenization_narrative.append((selection[0], tokens))
			full_normalized_query[selection[0]]["single_term_positional_index_narrative"] = tokens

		#now create single term tokenization aka now remove stop words
		#now create stem tokenization aka stem all words from single term index	

		for element in single_term_positional_tokenization:

			by_stop_words = [x for x in element[1] if x not in self.stop_words]
			stemmed = [stemmer.stem(thing) for thing in by_stop_words]

			single_term_tokenization.append((element[0], by_stop_words))
			full_normalized_query[element[0]]["single_term_index"] = by_stop_words
			full_normalized_query[element[0]]["stem_index"] = stemmed


		for nitem in single_term_positional_tokenization_narrative:

			by_stop_words = [x for x in nitem[1] if x not in self.stop_words]
			stemmed = [stemmer.stem(thing) for thing in by_stop_words]

			single_term_tokenization_narrative.append((nitem[0], by_stop_words))
			full_normalized_query[nitem[0]]["single_term_index_narrative"] = by_stop_words
			full_normalized_query[nitem[0]]["stem_index_narrative"] = stemmed


		#now make phrase tokenization for query aka use single 
		phrase_tokenization = [(j[0], self.find_phrases(j[1])) for j in phrase_normalized]
		phrase_tokenization_narrative = [(j[0], self.find_phrases(j[1])) for j in phrase_normalized_narrative]
		for dongxi in phrase_tokenization:
			full_normalized_query[dongxi[0]]["phrase_index"] = dongxi[1]
		for leaf in phrase_tokenization_narrative:
			full_normalized_query[leaf[0]]["phrase_index_narrative"] = leaf[1]


		self.index_query_terms()
		self.write_queries()

	def get_tf(self, index, term, document):

		term_id = -1
		tf = 0

		idx = self.indices[index]

		if term in idx.lexicon.keys():
			term_id = idx.lexicon[term]

			if term_id != -1:

				match = [tupl for tupl in idx.inverted_index[term_id][1] if tupl[0] == document]
				if match != []:
					tf  = match[0][1]

		return int(tf)


	#assumption is id exists in ii 
	def get_tf_with_term_id(self, index, term_id, document):

		tf = 0
		idx = self.indices[index]

		match = [tupl for tupl in idx.inverted_index[term_id][1] if tupl[0] == document]
		if match != []:
			tf  = match[0][1]

		return int(tf)


	def get_all_docs_for_term_id(self, index, term_id):

		posting_list = self.indices[index].inverted_index[term_id][1]

		all_docs = [tupl[0] for tupl in posting_list]

		return all_docs

	def get_all_docs_for_term(self, index, term):

		term_id = -1

		idx = self.indices[index]

		if term in idx.lexicon.keys():
			term_id = idx.lexicon[term]

		if term_id != -1:
			return self.get_all_docs_for_term_id(index, term_id)

		return []

	def get_all_terms_for_doc(self, index, document):

		result = []
		idx = self.indices[index]

		result = idx.doc_term_id_index[document][1]

		# result = [k for k in idx.inverted_index.keys() if document in self.get_all_docs_for_term_id(index, k)]

		return result

	def get_term_by_term_id(self, index, term_id):

		result = [thing[0] for thing in self.indices[index].lexicon.items() if thing[1] == term_id]
		return result[0]

	def get_df(self, index, term):

		term_id = -1
		df = 0

		idx = self.indices[index]

		if term in idx.lexicon.keys():
			term_id = idx.lexicon[term]

			if term_id != -1:
				df = idx.inverted_index[term_id][0]

		return int(df)

	def get_df_by_term_id(self, index, term_id):


		idx = self.indices[index]

		df = idx.inverted_index[term_id][0]

		return int(df)

	def compute_idf_vs(self, df):

		return math.log(self.total_documents/df, 10)


	#input requires getting all terms in a doc, this might costly since the only way would be to search the entire ii
	#function needs term frequency and document frequency per query term
	def normalized_tf_idf_doc(self, index, term, document):

		result = 0

		all_term_in_doc = self.get_all_terms_for_doc(index, document)
		tf = self.get_tf(index, term, document)
		df = self.get_df(index, term)

		if df != 0:
			idf = self.compute_idf_vs(df)

			top = idf * (math.log(tf+1, 10))

			bottom_sum = 0

			for each in all_term_in_doc:
				df = self.get_df_by_term_id(index, each)
				idf = self.compute_idf_vs(df)
				tf = self.get_tf_with_term_id(index, each, document)
				bottom_sum += (idf * (math.log(tf+1, 10)))**2

			result = top/bottom_sum

		return result

	def normalized_tf_idf_query(self, query_term_tf, query_term_df, query_dfs, query):

		result = 0
		
		if query_term_df != 0:

			idf = self.compute_idf_vs(query_term_df)
			
			top = idf * (math.log(query_term_tf +1, 10))

			bottom_sum = 0

			for idx, each in enumerate(query.keys()):
				tf = query[each]
				df = query_dfs[idx]
				if df != 0:
					idf = self.compute_idf_vs(df)
					bottom_sum += (idf * (math.log(tf+1, 10)))**2


			result = top/bottom_sum

		return result

	def calculate_cosine_similarity(self, index, query):
		#query is a dictionary of tokens with tf
		#document is a doc_id
		#index is idex type

		#get documents with query terms in them 
		docs_q_term = self.get_docs_for_query(index, query.keys())

		#now come up with a score for each document against the query
		#first the unnormalized score is the dot product of the query and document

		#each retrieved doc and its query term weight vector
		retrieved_doc_weight_vector = []
		for dc in docs_q_term:
			doc_weights = []
			for q in query.keys():
				doc_weights.append(self.normalized_tf_idf_doc(index, q, dc))
			retrieved_doc_weight_vector.append((dc, doc_weights))

		#now make the query term weight vector

		df_for_query_term  = [self.get_df(index, q_term) for q_term in query.keys()]
		query_term_weights = [self.normalized_tf_idf_query(tf, df_for_query_term[idx], df_for_query_term, query) for idx, tf in enumerate(query.values())]

		#dot product of query weight vector with each document

		unnormalized_dot_product = []

		for tupl in retrieved_doc_weight_vector:
			dot = sum([i*j for (i, j) in zip(tupl[1], query_term_weights)])
			unnormalized_dot_product.append((tupl[0], dot))

		#calculate divisor for normalizing cosine similarity

		normalization_doc_divisors = []
		summed_term_weights = sum([tw**2 for tw in query_term_weights])
		for item in retrieved_doc_weight_vector:
			summed_doc_weights = sum([weight**2 for weight in item[1]])
			normalization_doc_divisors.append((item[0], math.sqrt(summed_term_weights*summed_doc_weights)))

		#calcualte final normalized cosine similarity score
		#i.e. divide unnormalized dot product by normalization divisor for each document

		final_cosine_similarity = {}
		for doc_idx in range(0, len(unnormalized_dot_product)):

			top = unnormalized_dot_product[doc_idx][1]
			bottom = normalization_doc_divisors[doc_idx][1]

			if bottom == 0:
				r = 0
			else:
				r = top/bottom

			final_cosine_similarity.update({unnormalized_dot_product[doc_idx][0]: r })

		return final_cosine_similarity
		#output is a dictionary of doc_id: cs score

	def calculate_idf_bm25(self, index, term):

		docs_with_term = len(self.get_all_docs_for_term(index, term))
		return math.log( (self.total_documents - docs_with_term + 0.5)/(docs_with_term + 0.5) , 10)

	def get_avg_doc_length(self, index):

		avg = 0
		doc_index = self.indices[index].doc_term_id_index
		summ = sum([tupl[0] for tupl in doc_index.values()])
		avg = summ/self.total_documents

		return avg

	def get_doc_length(self, index, doc):

		doc_index = self.indices[index].doc_term_id_index
		return doc_index[doc][0]


	def calculate_bm25(self, index, query):

		avg_doc_length = self.get_avg_doc_length(index)

		#calculate query terms weights, just tf in this case
		query_terms = query.keys()
		
		#w is just the idf
		#then get all documents with query terms
		docs_q_term = self.get_docs_for_query(index, query_terms)

		doc_scores = {key:0 for key in docs_q_term}

		#then calculate doc term weights(tf) for each doc

		for doc in docs_q_term:

			summ = 0

			for qry in query_terms:
				
				doc_term_tf = self.get_tf(index, qry, doc)
				doc_length = self.get_doc_length(index, doc)
				K = self.K1*(1-self.B+self.B*(doc_length/avg_doc_length))
				w = self.calculate_idf_bm25(index, qry)
				first_factor = ((self.K1+1)*doc_term_tf)/(doc_term_tf + K)
				second_factor = ((self.K2 +1)*query[qry])/(self.K2 + query[qry])

				qry_result = w*first_factor*second_factor

				summ += qry_result

			doc_scores[doc] = summ

		return doc_scores

	def get_tf_in_collection(self, index, term):

		term_id = -1
		tf = 0

		idx = self.indices[index]
		if term in idx.lexicon.keys():
			term_id = idx.lexicon[term]

		if term_id != -1:
			pl = idx.inverted_index[term_id][1]
			tf = sum([int(tpl[1]) for tpl in pl])

		return tf

	def get_collection_size(self,index):

		size = 0

		idx = self.indices[index]

		for term_id in idx.lexicon.values():
			
			pl = idx.inverted_index[term_id][1]

			term_count_in_collection = sum([int(tpl[1]) for tpl in pl])

			size += term_count_in_collection

		return size

	def get_docs_for_query(self, index, query_terms):

		docs_q_term = set([])

		if index != "single_term_positional_index":
			for q in query_terms:
				docs = self.get_all_docs_for_term(index, q)
				for d in docs:
					docs_q_term.add(d)
		else:
			docs_q_term = self.determine_positional_index_docs(index, query_terms)

		return docs_q_term

	def calculate_kl_divergence(self, index, query):

		mu = self.get_avg_doc_length(index)
		if self.c_size[index] == None:
			self.c_size[index] = self.get_collection_size(index)

		collection_size = self.c_size[index]

		#calculate query terms weights, just tf in this case
		query_terms = query.keys()
		
		#w is just the idf
		#then get all documents with query terms
		docs_q_term = self.get_docs_for_query(index, query_terms)

		doc_scores = {key:0 for key in docs_q_term}

		#create query model once
		query_model = {}
		query_length = float(sum(query.values()))

		for term in query_terms:
			query_model.update({term:query[term]/query_length})

		#create document scores
		for doc in docs_q_term:

			summ = 0

			for qry in query_terms:


				dirichlet_smoothing_top = self.get_tf(index, qry, doc) + mu*(float(self.get_tf_in_collection(index, qry))/collection_size)
				dirichlet_smoothing_bottom = self.get_doc_length(index, doc) + mu

				dirichlet_smoothing_val = math.log( dirichlet_smoothing_top/dirichlet_smoothing_bottom + 0.0001, 10)

				result = dirichlet_smoothing_val * query_model[qry]

				summ += result

			doc_scores[doc] = summ

		return doc_scores

	def get_position_list(self, index, term, doc):

		idx = self.indices[index]

		posting_list = idx.inverted_index[idx.lexicon[term]][1]

		sought_tuple = [t for t in posting_list if t[0] == doc][0]

		position_list = sought_tuple[2]

		return position_list

	def determine_positional_index_docs(self, index, query):

		idx = self.indices[index]
		relevant_docs = set([])
		phrase_count = 0

		query_doc_tupl = [(key,self.get_all_docs_for_term(index, key)) for key in query]

		for q_term_index in range(0,len(query_doc_tupl)-1):#go through each query term
			q_term_doc_list = query_doc_tupl[q_term_index][1]#doc list for current term
			q_term = query_doc_tupl[q_term_index][0]#current term
			if q_term in idx.lexicon.keys():
				next_query_term = query_doc_tupl[q_term_index+1][0]
				if next_query_term in idx.lexicon.keys():
					doc_list_next_term = query_doc_tupl[q_term_index+1][1] 
					for doc in q_term_doc_list:#for each doc for current term
						phrase_count = 0
						if doc in doc_list_next_term:#check if next term exists in the same doc, if does not then a phrase is impossible
							position_list = self.get_position_list(index, q_term, doc)#get position list for current term
							position_list_next_term = self.get_position_list(index, next_query_term, doc)
							for position in position_list:#see if position + 1 exists in the next term's doc
								if position +1 in position_list_next_term:
									phrase_count += 1
						if phrase_count > 0:
							relevant_docs.add(doc)

		return relevant_docs

	def prepare_initial_results(self, query_num, ranking_dictionary):

		#get list of tuples doc_id:similarity score sorted in descending order, to the documents are ranked
		descending_score_tuples = sorted(ranking_dictionary.items(), key=operator.itemgetter(1), reverse = True)
		#add the list to results list under the query number key
		self.results.append((query_num, descending_score_tuples[:100]))

	def prepare_reformulated_results(self, query_num, ranking_dictionary):

		#get list of tuples doc_id:similarity score sorted in descending order, to the documents are ranked
		descending_score_tuples = sorted(ranking_dictionary.items(), key=operator.itemgetter(1), reverse = True)
		#add the list to results list under the query number key
		self.reformulated_results.append((query_num, descending_score_tuples[:100]))


	#outputs results doc in format for treceval.exe
	def write_results_file(self):

		#sort the results list by the query key resulting in ascending order
		self.results.sort(key=lambda tup: tup[0])
		
		results_file = open(RESULTS_FILE, 'w')

		#write results to file
		for item in self.results:
			#write query num

			for idx, r in enumerate(item[1]):
				
				# query num, 0, doc_id, rank, score
				results_file.write(str(item[0]) + "\t0\t" + r[0] + "\t" + str(idx) + "\t  ")
				results_file.write(str(r[1]))
				results_file.write("\t" + "comment"+ "\n")

		results_file.close()

		if self.rank_method == "cosine_similarity":
			REFORM_RESULTS_FILE = VS_REFORM_RESULTS_FILE

		elif self.rank_method == "bm25":
			REFORM_RESULTS_FILE = BM25_REFORM_RESULTS_FILE

		elif self.rank_method == "kl_divergence":
			REFORM_RESULTS_FILE = KL_REFORM_RESULTS_FILE

		reform_results_file = open(REFORM_RESULTS_FILE, 'w')

		for item in self.reformulated_results:
			#write query num

			for idx, r in enumerate(item[1]):
				
				# query num, 0, doc_id, rank, score
				reform_results_file.write(str(item[0]) + "\t0\t" + r[0] + "\t" + str(idx) + "\t  ")
				reform_results_file.write(str(r[1]))
				reform_results_file.write("\t" + "comment"+ "\n")


		reform_results_file.close()

	def expand_query(self, index, results, query_terms):

		n_dict = {}
		reform_query = copy.deepcopy(query_terms)

		descending_score_tuples = sorted(results.items(), key=operator.itemgetter(1), reverse = True)
			
		top_n_docs = descending_score_tuples[0:self.top_n_docs]

		for doc in top_n_docs:
			doc_terms = self.get_all_terms_for_doc(index, doc[0])
			for term in doc_terms:
				idf = self.compute_idf_vs(self.get_df_by_term_id(index, term))
				if term in n_dict.keys():
					n_dict[term] += 1 * idf
				else:
					n_dict.update({term:1 * idf})

		descending_term_weigths = sorted(n_dict.items(), key=operator.itemgetter(1), reverse = True)

		terms = [self.get_term_by_term_id(index, t_id[0]) for t_id in  descending_term_weigths[0:self.top_t_terms]]

		if index != "single_term_positional_index":

			reform_query.update(terms)

		elif index == "single_term_positional_index":

			for t in terms:
				if t in reform_query.keys():
					reform_query[t] += 1
				else:
					reform_query.update({t:1})

		return reform_query

	def reduce_query(self, index, query_terms):

		n_dict = {}

		#use tf_q * idf

		for k in query_terms.keys():

			df = self.get_df(index, k)
			if df == 0:
				df = 0.0001

			n_dict.update({k:query_terms[k]*self.compute_idf_vs(df)})

		descending_score_tuples = sorted(n_dict.items(), key=operator.itemgetter(1), reverse = True)

		top_terms = [t[0] for t in descending_score_tuples[0:self.query_term_threshold]]
		# print tokens

		# x= [ z.split("\"") for z in top_terms]
		# top_terms = [thing for sublist in tokens for thing in sublist]


		# if index != "single_term_positional_index":

		reform_query = Counter()
		reform_query.update(top_terms)

		# elif index == "single_term_positional_index":

		# 	reform_query = OrderedDict()
		# 	for t in top_terms:
		# 		if t in reform_query.keys():
		# 			reform_query[t] += 1
		# 		else:
		# 			reform_query.update({t:1})

		return reform_query

	def rank(self, index, rank_function, query):

		search_results = rank_function(index, query)
		return search_results

	def reformulate(self, index, search_results, query):

		if self.reduc_expan == "expansion":

			reform = self.expand_query(index, search_results, query)

		elif self.reduc_expan == "reduction":

			reform = self.reduce_query(index, query)

		print "original query is: " + " ".join(query.keys())
		print "reformulated query is: " + " ".join(reform.keys())
		return reform

	def get_map_rel_ret(self):

		if self.rank_method == "cosine_similarity":
			REFORM_RESULTS_FILE = "cosine_reform_results.txt"

		elif self.rank_method == "bm25":
			REFORM_RESULTS_FILE = "bm25_reform_results.txt"

		elif self.rank_method == "kl_divergence":
			REFORM_RESULTS_FILE = "kl_reform_results.txt"


		trec_eval_output = subprocess.check_output([TREC_EVAL_FILE, "-a", "qrel2.txt", REFORM_RESULTS_FILE])
		map_line = trec_eval_output.split("\n")[4]
		rel_ret_line = trec_eval_output.split("\n")[3]
		map_score =  float(map_line.split("\t")[2])
		rel_ret = int(rel_ret_line.split("\t")[2])

		return map_score, rel_ret

	def search_and_rank(self):

		rank_function = None
		search_results = {}
		index_used = self.default_index
		reformulated_results = {}

		#set the rank function
		if self.rank_method == "cosine_similarity":
			rank_function = self.calculate_cosine_similarity
		elif self.rank_method == "bm25":
			rank_function = self.calculate_bm25
		elif self.rank_method == "kl_divergence":
			rank_function = self.calculate_kl_divergence


		if self.reduc_expan == "expansion":

			for idx, each in enumerate(self.full_normalized_query.keys()):

				print "query number: " + each
				print "query " + str(idx+1) + "/" + str(len(self.full_normalized_query.keys()))

				#check phrase

				if self.no_phrases == False:

					phrase_docs = len(self.get_docs_for_query("phrase_index", self.full_normalized_query[each]["phrase_index"]))

					if phrase_docs >= self.PHRASE_THRESHOLD:

						index_used = "phrase_index"
						print "passed phrase threshold of " + str(self.PHRASE_THRESHOLD) + " found " +str(phrase_docs) + ", scoring on phrase index"				
						
						#search_results, reformulated_results = self.ranking("phrase_index", rank_function, self.full_normalized_query[each]["phrase_index"])
						search_results = self.rank(index_used, rank_function, self.full_normalized_query[each][index_used])
						reformulated_query = self.reformulate(index_used, search_results, self.full_normalized_query[each][index_used])
						reformulated_results = self.rank(index_used, rank_function, reformulated_query)

					else:

						#check positional
						positional_docs = len(self.get_docs_for_query("single_term_positional_index", self.full_normalized_query[each]["single_term_positional_index"]))

						if positional_docs >= self.POSITIONAL_TRESHOLD:

							index_used = "single_term_positional_index"
							print "passed positional threshold of " + str(self.POSITIONAL_TRESHOLD) + " found " +str(positional_docs) + ", scoring on positional index"
							
							# search_results, reformulated_results = self.ranking("single_term_positional_index", rank_function, self.full_normalized_query[each]["single_term_positional_index"])
							search_results = self.rank(index_used, rank_function, self.full_normalized_query[each][index_used])
							reformulated_query = self.reformulate(index_used, search_results, self.full_normalized_query[each][index_used])
							reformulated_results = self.rank(index_used, rank_function, reformulated_query)

						else:

							index_used = self.default_index
							#choose between single term and stem index, base decision on which index returns more documents
							default_index_docs = len(self.get_docs_for_query(self.default_index, self.full_normalized_query[each][self.default_index]))

							print "using defualt index " + self.default_index + " with retrieved docs of count " + str(default_index_docs)
							
							# search_results, reformulated_results = self.ranking(self.default_index, rank_function, self.full_normalized_query[each][self.default_index])
							search_results = self.rank(index_used, rank_function, self.full_normalized_query[each][index_used])
							reformulated_query = self.reformulate(index_used, search_results, self.full_normalized_query[each][index_used])
							reformulated_results = self.rank(index_used, rank_function, reformulated_query)

				else:

					index_used = self.default_index

					default_index_docs = len(self.get_docs_for_query(self.default_index, self.full_normalized_query[each][self.default_index]))

					print "using defualt index " + self.default_index + " with retrieved docs of count " + str(default_index_docs)
					# search_results, reformulated_results = self.ranking(self.default_index, rank_function, self.full_normalized_query[each][self.default_index])
					search_results = self.rank(index_used, rank_function, self.full_normalized_query[each][index_used])
					reformulated_query = self.reformulate(index_used, search_results, self.full_normalized_query[each][index_used])
					reformulated_results = self.rank(index_used, rank_function, reformulated_query)

				print "\n\n\n"

				self.prepare_initial_results(each, search_results)
				self.prepare_reformulated_results(each, reformulated_results)

		else:#self.reduc_expan == "reduction"

			for idx, each in enumerate(self.full_normalized_query.keys()):

				print "query number: " + each
				print "query " + str(idx+1) + "/" + str(len(self.full_normalized_query.keys()))

				#check phrase

				if self.no_phrases == False:

					phrase_docs = len(self.get_docs_for_query("phrase_index", self.full_normalized_query[each]["phrase_index"+"_narrative"]))

					if phrase_docs >= self.PHRASE_THRESHOLD:

						index_used = "phrase_index"
						print "passed phrase threshold of " + str(self.PHRASE_THRESHOLD) + " found " +str(phrase_docs) + ", scoring on phrase index"				
						
						#search_results, reformulated_results = self.ranking("phrase_index", rank_function, self.full_normalized_query[each]["phrase_index"])
						
						if self.reduc_orig == True:
							search_results = self.rank(index_used, rank_function, self.full_normalized_query[each][index_used+"_narrative"])
						reformulated_query = self.reformulate(index_used, search_results, self.full_normalized_query[each][self.default_index+"_narrative"])
						reformulated_results = self.rank(self.default_index, rank_function, reformulated_query)

					else:

						#check positional
						positional_docs = len(self.get_docs_for_query("single_term_positional_index", self.full_normalized_query[each]["single_term_positional_index"+"_narrative"]))

						if positional_docs >= self.POSITIONAL_TRESHOLD:

							index_used = "single_term_positional_index"
							print "passed positional threshold of " + str(self.POSITIONAL_TRESHOLD) + " found " +str(positional_docs) + ", scoring on positional index"
							
							# search_results, reformulated_results = self.ranking("single_term_positional_index", rank_function, self.full_normalized_query[each]["single_term_positional_index"])
							if self.reduc_orig == True:
								search_results = self.rank(index_used, rank_function, self.full_normalized_query[each][index_used+"_narrative"])
							reformulated_query = self.reformulate(index_used, search_results, self.full_normalized_query[each][self.default_index+"_narrative"])
							reformulated_results = self.rank(self.default_index, rank_function, reformulated_query)

						else:

							index_used = self.default_index
							#choose between single term and stem index, base decision on which index returns more documents
							default_index_docs = len(self.get_docs_for_query(self.default_index, self.full_normalized_query[each][self.default_index+"_narrative"]))

							print "using defualt index " + self.default_index + " with retrieved docs of count " + str(default_index_docs)
							
							# search_results, reformulated_results = self.ranking(self.default_index, rank_function, self.full_normalized_query[each][self.default_index])
							if self.reduc_orig == True:
								search_results = self.rank(index_used, rank_function, self.full_normalized_query[each][index_used+"_narrative"])
							reformulated_query = self.reformulate(index_used, search_results, self.full_normalized_query[each][index_used+"_narrative"])
							reformulated_results = self.rank(index_used, rank_function, reformulated_query)

				else:

					index_used = self.default_index

					default_index_docs = len(self.get_docs_for_query(self.default_index, self.full_normalized_query[each][self.default_index+"_narrative"]))

					print "using defualt index " + self.default_index + " with retrieved docs of count " + str(default_index_docs)
					# search_results, reformulated_results = self.ranking(self.default_index, rank_function, self.full_normalized_query[each][self.default_index])
					
					if self.reduc_orig == True:
						search_results = self.rank(index_used, rank_function, self.full_normalized_query[each][index_used+"_narrative"])
					reformulated_query = self.reformulate(index_used, search_results, self.full_normalized_query[each][index_used+"_narrative"])
					reformulated_results = self.rank(index_used, rank_function, reformulated_query)


				print "\n\n\n"


				self.prepare_initial_results(each, search_results)
				self.prepare_reformulated_results(each, reformulated_results)

	def run(self):

		#normalizes all the query terms
		self.process()#processed queries are stored in self.full_normalized_query dictionary

		#tokenize each query four different ways, 
		#so each query has 4 different tokinzations able to be used by each of the different indices

		if self.test == True:

			top_docs = []
			top_terms = []
			map_scores =[]
			rel_rets = []

			done = False

			if self.reduc_expan == "expansion":

				self.top_n_docs = 5
				self.top_t_terms = 5

				while done != True:

					print "top_n_docs " + str(self.top_n_docs)
					print "top_t_terms " + str(self.top_t_terms)

					self.results = []
					self.reformulated_results = []

					self.search_and_rank()
					self.write_results_file()
					map_score, rel_ret = self.get_map_rel_ret()

					top_docs.append(self.top_n_docs)
					top_terms.append(self.top_t_terms)
					map_scores.append(map_score)
					rel_rets.append(rel_ret)

					if self.top_n_docs ==1 and self.top_t_terms == 1:
						done = True

					if self.top_t_terms != 1:
						self.top_t_terms -= 1

					elif self.top_n_docs != 1:
						self.top_n_docs -= 1
						self.top_t_terms = 5


				f = open(CWD+"/"+self.rank_method+"_test_results.csv", "wt")
				writer = csv.writer(f)

				writer.writerow(top_docs)
				writer.writerow(top_terms)
				writer.writerow(map_scores)
				writer.writerow(rel_rets)

				f.close()

			elif self.reduc_expan == "reduction":

				print "testing query reduction"

				self.query_term_threshold = 6

				while done != True:

					print "query_term_threshold " + str(self.query_term_threshold)
					
					self.results = []
					self.reformulated_results = []

					self.search_and_rank()
					self.write_results_file()
					map_score, rel_ret = self.get_map_rel_ret()

					top_terms.append(self.query_term_threshold)
					map_scores.append(map_score)
					rel_rets.append(rel_ret)

					if self.query_term_threshold != 3:
						self.query_term_threshold -= 1
					else:
						done = True


				f = open(CWD+"/"+self.rank_method+"_test_results_reduc.csv", "wt")
				writer = csv.writer(f)

				writer.writerow(top_terms)
				writer.writerow(map_scores)
				writer.writerow(rel_rets)

				f.close()


		else:

			self.search_and_rank()
			self.write_results_file()




def get_queries():

	queries = open(QUERY_FILE, 'r')
	query_list = []
	lines = []
	current_number = None
	current_title = None
	top = False
	current_narrative = ""

	for line in queries:
		if line != "\n":
			lines.append(line.strip("\n"))

	length = len(lines)
	idx = 0

	while idx < length:

		current_line = lines[idx]
		
		if '<top>' in current_line or top == True:

			top = True

			if '<num>' in current_line:

				current_number = current_line


			elif '<title>' in current_line:

				current_title = current_line
				# print current_line


			elif '<narr>' in current_line:

				current_narrative = ''
				bottom = False

				while bottom == False:

					idx += 1
					current_line = lines[idx]

					if '</top>' in current_line:
						bottom = True
					else:
						current_narrative += current_line + " "


		if '</top>' in current_line:

			final_number = current_number.split(":")[1].strip().strip("\n")
			final_title = current_title.split(":")[1].strip().strip("\n")

			query_list.append(Query(final_number, final_title, current_narrative))

			top = False
			current_number = None
			current_title = None
			current_narrative = ""


		idx += 1

	return query_list

def calculate_doc_length(index_obj, doc_index, doc):

	pl = doc_index[doc]

	length = 0

	for term_id in pl:

		term_pl = index_obj.inverted_index[term_id][1]

		tf_for_doc = sum([int(tupl[1]) for tupl in term_pl if tupl[0] == doc])

		length += tf_for_doc

	return length


def write_doc_index(index_type, index_obj):

	doc_index = index_obj.doc_term_id_index

	wfile = open(INDICES_PATH + index_type + "_doc_term_index.txt", 'w')

	for key in doc_index.keys():

		doc_length = calculate_doc_length(index_obj, doc_index, key)

		wfile.write(str(key) + " " + str(doc_length) + " ")

		for item in doc_index[key]:

			wfile.write(str(item) + ",")

		wfile.write("\n")

	wfile.close()


def load_index(index_type):

	inverted_index = {}
	doc_term_id_index = {}
	lexicon = {}

	if index_type == "single_term_index":
		lfile = open(SINGLE_TERM_LEXICON, 'r')
		ifile = open(SINGLE_TERM_II, 'r')
		dfile = open(SINGLE_TERM_DOC_INDEX, 'r')
	if index_type == "single_term_positional_index":
		lfile = open(SINGLE_TERM_POSITIONAL_LEXICON, 'r')
		ifile = open(SINGLE_TERM_POSITIONAL_II, 'r')
		dfile = open(SINGLE_TERM_POSITIONAL_DOC_INDEX, 'r')
	if index_type == "stem_index":
		lfile = open(STEM_LEXICON, 'r')
		ifile = open(STEM_II, 'r')
		dfile = open(STEM_DOC_INDEX, 'r')
	if index_type == "phrase_index":
		lfile = open(PHRASE_LEXICON, 'r')
		ifile = open(PHRASE_II, 'r')
		dfile = open(PHRASE_DOC_INDEX, 'r')

	#all dfiles are the same

	for hang in dfile:
		hang = hang.strip("\n")

		splits = hang.split(" ")
		doc_id = splits[0]
		doc_length = int(splits[1])
		term_list = splits[2]

		memory_pl = []

		for term in term_list.split(","):

			if term != "":

				memory_pl.append(int(term))

		doc_term_id_index.update({doc_id:(doc_length,memory_pl)})


	#only phrase lexicon has different format
	if index_type != "phrase_index":

		for line in lfile:

			splits = line.split(" ")
			lexicon.update({splits[0].decode('utf-8'):int(splits[1].strip("\n"))})

	else:

		for line in lfile:

			splits = line.split(" ")
			phrase = " ".join(splits[:len(splits)-1])
			lexicon.update({phrase.decode('utf-8'):int(splits[len(splits)-1].strip("\n"))})

	#only single term positional index has different format
	if index_type != "single_term_positional_index":

		for row in ifile:

			row_splits = row.split(" ")
			termid = int(row_splits[0])
			df = int(row_splits[1])

			posting_list = row_splits[2:]

			parsed_posting_list = map(lambda s: (s.split(",")[0], s.split(",")[1].strip("\n")), posting_list)


			df_posting_list = (df, parsed_posting_list)

			inverted_index.update({termid:df_posting_list})

	else:

		for row in ifile:

			row_splits = row.split(" ")
			termid = int(row_splits[0])
			df = int(row_splits[1])

			posting_list = row_splits[2:]

			position_list = []
			parsed_posting_list = []

			for item in posting_list:

				docid, tf, positions = item.split(",")
				positions = positions.strip("\n").strip("{").strip("}")

				position_list = positions.split("_")[1:]
				position_list = map(lambda s: int(s), position_list)

				parsed_posting_list.append((docid, int(tf), position_list))


			df_posting_list = (df, parsed_posting_list)

			inverted_index.update({termid:df_posting_list})

	index = Index(lexicon, inverted_index, doc_term_id_index)

	# write_doc_index(index_type, index)

	return index

def setup(command_args):

	rank_methods = ["cosine_similarity", "bm25", "kl_divergence"]
	indices = ["single_term_index", "single_term_positional_index", "stem_index", "phrase_index"]
	reformulation = ["reduction", "expansion"]

	rank_method = "cosine_similarity"
	default_index = "single_term_index"
	no_phrases = False
	r_e = "expansion"
	test = False
	reduc_orig = False

	rank = [k for k in sys.argv if k in rank_methods]

	if len(rank) != 0:
		if rank[0] == "bm25":
			rank_method = "bm25"

		elif rank[0] == "kl_divergence":
			rank_method = "kl_divergence"

	stem = [1 for k in sys.argv if k == "stem_index"]

	if len(stem) != 0:
		default_index = "stem_index"

	
	x = [1 for k in sys.argv if k == "True"]
	if len(x) != 0: 
		no_phrases = True


	if_re = [1 for k in sys.argv if k == "reduction"]
	if len(if_re) != 0:
		r_e = "reduction"

	rdo = [1 for k in sys.argv if k == "reduc_orig"]
	if len(rdo) != 0:
		reduc_orig = True


	if_test = [1 for k in sys.argv if k == "-test"]
	if len(if_test) != 0:
		test= True

	print "ranking query results by " + rank_method
	print "default index: " + default_index
	print "no_phrases set to " + str(no_phrases)
	print "reduction or expansion: " + r_e
	qp = Query_Processor(rank_method, default_index, no_phrases, r_e, test, reduc_orig,get_queries(), single_term_index = load_index("single_term_index"), single_term_positional_index = load_index("single_term_positional_index"), stem_index = load_index("stem_index"), phrase_index = load_index("phrase_index"))

	return qp

def main():

	start = time.time()
	print "setting up"
	
	qp = setup(sys.argv)
	end_processing = time.time() - start

	print "processing"
	qp.run()

	end_retrieval = time.time() - start

	print "processing query text time: " + str(end_processing) + " seconds"
	print "retrieving search results time: " + str(end_retrieval/60) + " minutes"
	print "total time: " + str((time.time() - start)/60) + " minutes"
	print "time per query: " + str((time.time()-start)/21) + " seconds"


	#findings
	#some "relevant" documents were not retrieved because only title terms were used and it appears the relevancy ratings included the entire query text
if __name__ == "__main__":
	main()


"""

spare code

			# docs = [tupl[0] for tupl in parsed_posting_list]#all docs with this term
			
			# for d in docs:
			# 	if d not in doc_term_id_index.keys():
			# 		doc_term_id_index.update({d:[]})
			# 	doc_term_id_index[d].append(termid)

"""