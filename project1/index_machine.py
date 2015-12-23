# coding= utf-8

import os
import re
import time
import datetime
import sys
from collections import Counter
from collections import OrderedDict
from collections import defaultdict
import heapq as h
from nltk.stem.porter import PorterStemmer

DOC_NAMES_FILE = 'doc_names.txt'

DOC_FILES_PATH = os.getcwd() + '/Mini-Trec-Data/BigSample/'
STOP_WORDS_PATH = os.getcwd() + '/Mini-Trec-Data/stops.txt'
MERGE_FILE_SIZE = 25


class Index_Machine:

	class Doc:

		def __init__(self, docid, name):
			self.doc_id = docid
			self.text_file_name = name
			self.doc_text = []

		def add_text(self, text):
			self.doc_text.append(text)

	class Index:

		def __init__(self):
			self.lexicon = {'alpha':{'a':[], 'b':[], 'c':[], 'd':[], 'e':[], 'f':[], 'g':[], 'h':[], 'i':[], 'j':[], 'k':[], 'l':[], 'm':[], 'n':[], 'o':[], 'p':[], 'q':[], 'r':[], 's':[], 't':[], 'u':[], 'v':[], 'w':[], 'x':[], 'y':[], 'z':[]}, 'numeral':[] } #make an alpha order dict
			self.lexicon_count = 0
			self.posting_list = []
			self.posting_list_count = 0
			self.current_doc = ""
			self.index_type = ""
			self.memory_restriction = ""
			self.inverted_index = {}

		def append_lexicon_single_term_positional(self, word_count_positions):

			word = word_count_positions[0]
			count = word_count_positions[1]
			positions = word_count_positions[2:]
			doc_id = self.current_doc
			first_char = word[0]
			word_termid = (word, self.lexicon_count)
			alpha = self.lexicon['alpha']
			nums = self.lexicon['numeral']
			ii = self.inverted_index

			if first_char in alpha.keys():

				letter_list = alpha[first_char]

				words_letter_list = [item[0] for item in letter_list if item[0] == word]

				if words_letter_list == []:#if term not in lexicon
					letter_list.append(word_termid)
					self.lexicon_count += 1

				#then make posting list entry

				#get termid

				#lexicon is in (word, id) format
				#only get the id
				term_id_retrieved = [thing[1] for thing in letter_list if thing[0] == word]

				t_id = term_id_retrieved[0]

				if self.memory_restriction == 'unlimited':

					if t_id not in ii.keys():
						ii.update({t_id:[]})

					ii[t_id].append((doc_id, count, tuple(positions)))

				else:

					posting_list_tuple = (t_id, doc_id, count) + tuple(positions) #key value pair
					self.posting_list.append(posting_list_tuple)

			else:#must be a number

				words_nums_list = [num[0] for num in nums if num[0] == word]

				if words_nums_list ==[]:#if the term is not in the lexicon
					nums.append(word_termid)
					self.lexicon_count += 1

				#then make posting list entry

				term_id_retrieved = [thing[1] for thing in nums if thing[0] == word]

				t_id = term_id_retrieved[0]

				if self.memory_restriction == 'unlimited':

					if t_id not in ii.keys():
						ii.update({t_id:[]})

					ii[t_id].append((doc_id, count, tuple(positions)))

				else:
					posting_list_tuple = (t_id, doc_id, count) + tuple(positions) #key value pair
					self.posting_list.append(posting_list_tuple)

			self.posting_list_count += 1

		def append_lexicon_single_term_stem_phrase_index(self, word_count_pair):

			word = word_count_pair[0]
			count = word_count_pair[1]
			doc_id = self.current_doc
			first_char= word[0]
			word_termid = (word, self.lexicon_count)
			alpha = self.lexicon['alpha']
			nums = self.lexicon['numeral']
			ii = self.inverted_index

			if first_char in alpha.keys():

				letter_list = alpha[first_char]

				words_letter_list = [item[0] for item in letter_list if item[0] == word]

				if words_letter_list == []:#if term not in lexicon
					letter_list.append(word_termid)
					self.lexicon_count += 1

				#then make posting list entry

				#get termid

				#lexicon is in (word, id) format
				#only get the id
				term_id_retrieved = [thing[1] for thing in letter_list if thing[0] == word]

				t_id = term_id_retrieved[0]
			
				if self.memory_restriction == 'unlimited':

					if t_id not in ii.keys():
						ii.update({t_id:[]})

					ii[t_id].append((doc_id,count))

				else:

					posting_list_tuple = (t_id, doc_id, count) #key value pair
					self.posting_list.append(posting_list_tuple)

			else:#must be a number

				words_nums_list = [num[0] for num in nums if num[0] == word]

				if words_nums_list ==[]:#if the term is not in the lexicon
					nums.append(word_termid)
					self.lexicon_count += 1

				#then make posting list entry

				term_id_retrieved = [thing[1] for thing in nums if thing[0] == word]

				t_id = term_id_retrieved[0]

				if self.memory_restriction == 'unlimited':

					if t_id not in ii.keys():
						ii.update({t_id:[]})
	
					ii[t_id].append((doc_id,count))

				else:

					posting_list_tuple = (t_id, doc_id, count) #key value pair
					self.posting_list.append(posting_list_tuple)

			self.posting_list_count += 1			


	def __init__(self, index_type = "single_term_index", memory_restriction = "unlimited"):
		self.doc_list = []
		self.index_type = index_type
		self.memory_restriction = memory_restriction
		self.index = self.Index()
		self.index.index_type = index_type
		self.index.memory_restriction = memory_restriction
		self.date_dict = {"january":1, "jan":1, "february":2, "feb":2, "march":3, "mar":3, "april":4, "apr":4, "may":5, "jun":6, "june":6, "july":7, "jul":7, "aug":8, "august":8, "sept":9, "september":9, "oct":10, "october":10, "nov":11, "november":11, "dec":12, "december":12}
		self.dates = []
		self.stop_words = []
		self.common_prefixes = ["un", "re", "in", "im", "ir", "ill", "dis", "en", "em", "non", "in", "im", "over", "mis", "sub", "pre", "inter", "fore", "de", "trans", "super", "semi", "anti", "mid", "under"]
		self.forbidden_char = ['-', '/', ',', '[', ']', '(', ')', '@', '^', '#', '!', '%', '&', '*', '?', '|', ';', '{', '}', ':', '`', '\'', '_', '=', '+', '<', '>', '\\', '.', '~']
		self.temp_file_names = []
		self.temp_file_count = 0
		self.time_stamp = ""


	def get_doc_list(self):
		return self.doc_list


	def take_tokens(self, tokens_list):

		if self.index_type == "single_term_index":
		#remove stop words

			tokens_list = [x for x in tokens_list if x not in self.stop_words]
			
			counted_doc_items = Counter(tokens_list)
			
			counted_tokens = [(word, counted_doc_items[word]) for word in counted_doc_items.keys()]

			
			map(self.index.append_lexicon_single_term_stem_phrase_index, counted_tokens)


		elif self.index_type == "single_term_positional_index":

			#keep the stop words
			#make positional index

			#count occurence in doc
			counted_doc_items = Counter(tokens_list)

			#get positions for word in doc
			word_position_pair = [(item,i) for i, j in enumerate(tokens_list) for item in counted_doc_items.keys() if j == item]

			#merge positions for same word

			dictionare = {}

			for k, v in word_position_pair:
				dictionare.setdefault(k, [k]).append(v)

			merged_word_positions = map(tuple, dictionare.values())

			#combine word count with word position

			word_count_positions = []

			for it in merged_word_positions:

				tupl = (it[0], counted_doc_items[it[0]])
				tupl2 = tuple(it[1:len(it)])

				word_count_positions.append(tupl+tupl2)


			map(self.index.append_lexicon_single_term_positional, word_count_positions)


		elif self.index_type == "stem_index":

			tokens_list = [x for x in tokens_list if x not in self.stop_words]
			
			stemmer = PorterStemmer()

			stemmed = [stemmer.stem(thing) for thing in tokens_list]

			counted_doc_items = Counter(stemmed)

			counted_tokens = [(word, counted_doc_items[word]) for word in counted_doc_items.keys()]

			map(self.index.append_lexicon_single_term_stem_phrase_index, counted_tokens)

		
		elif self.index_type == "phrase_index":

			counted_doc_items = Counter(tokens_list)
			
			counted_tokens = [(word, counted_doc_items[word]) for word in counted_doc_items.keys()]

			map(self.index.append_lexicon_single_term_stem_phrase_index, counted_tokens)


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

	def normalize(self, doc):

		#make all lowercase
		doc.doc_text = map(lambda strng: strng.lower(), doc.doc_text)
		
		#make into one string in list
		one_string = " ".join(doc.doc_text)
		doc.doc_text = [one_string]

		#redo unicode
		doc.doc_text[0] = doc.doc_text[0].replace("&racute;", u"ŕ")
		doc.doc_text[0] = doc.doc_text[0].replace("&atilde;", u"ã")
		doc.doc_text[0] = doc.doc_text[0].replace("&hyph;", u"-")
		doc.doc_text[0] = doc.doc_text[0].replace("&eacute", u"é")
		doc.doc_text[0] = doc.doc_text[0].replace("&ntilde;", u"ñ")
		doc.doc_text[0] = doc.doc_text[0].replace("&agrave;", u"à")
		doc.doc_text[0] = doc.doc_text[0].replace("&oacute;", u"ó")
		doc.doc_text[0] = doc.doc_text[0].replace("&egrave;", u"è")
		doc.doc_text[0] = doc.doc_text[0].replace("&ocirc;", u"ô")
		doc.doc_text[0] = doc.doc_text[0].replace("&aacute;", u"á")
		doc.doc_text[0] = doc.doc_text[0].replace("&uuml;", u"ü")
		doc.doc_text[0] = doc.doc_text[0].replace("&rsquo;", u"'")
		doc.doc_text[0] = doc.doc_text[0].replace("&iacute;", u"í")
		doc.doc_text[0] = doc.doc_text[0].replace("&cent;", u"¢")
		doc.doc_text[0] = doc.doc_text[0].replace("&ccedil;", u"ç")

		#remove other ampersand strings
		ampersand_char = re.findall("&\w+;", doc.doc_text[0])
		ampersand_set = set(ampersand_char)

		for item in ampersand_set:
			doc.doc_text[0] = doc.doc_text[0].replace(item, " ")

		if self.index_type != 'phrase_index':
			#remove characters that have no use, even in speical tokens
			doc.doc_text[0] = doc.doc_text[0].replace("<", " ")
			doc.doc_text[0] = doc.doc_text[0].replace(">", " ")
			doc.doc_text[0] = doc.doc_text[0].replace("\n", " ")



			#normalizes the dates for a few date formats
			dates = []
		
			#find all dates in doc_text
			mmddyyyy_yy= re.findall("(\d+[-\\/]\d+[-\\/]\d+)\D", doc.doc_text[0])
			mmmddyyyy = re.findall("((?:jan|feb|mar|apr|jun|jul|aug|sept|oct|nov|dec)[,\\.\s-]+\d+[\s,-]+\d+)\D", doc.doc_text[0])
			nameddyyyy = re.findall("((?:january|february|march|april|may|june|july|august|september|october|november|december)\s+\d+(?:,|\s)+\d+)\D", doc.doc_text[0])
			ddnameyyyy = re.findall("\D(\d{1,2}[-\s]+(?:january|february|march|april|may|june|july|august|september|october|november|december|jan|feb|mar|apr|jun|jul|aug|sept|oct|nov|dec)[-\\.,\s]+\d+)[^0-9]", doc.doc_text[0])

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

					doc.doc_text[0] = doc.doc_text[0].replace(key, converted_mmddyyyy_yy[key])
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

					doc.doc_text[0] = doc.doc_text[0].replace(key, converted_ddnameyyyy[key])
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

					doc.doc_text[0] = doc.doc_text[0].replace(key, converted_mmmddyyyy[key])
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

					doc.doc_text[0] = doc.doc_text[0].replace(key, converted_nameddyyyy[key])
					dates.append(converted_nameddyyyy[key])


			self.dates = [y for y in dates if y != ""]

		return doc

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

	def normalize_phrase(self, doc):

		#make all lowercase
		doc.doc_text = map(lambda strng: strng.lower(), doc.doc_text)
		
		#make into one string in list
		one_string = " ".join(doc.doc_text)
		doc.doc_text = [one_string]

		#redo unicode
		doc.doc_text[0] = doc.doc_text[0].replace("&racute;", u"ŕ")
		doc.doc_text[0] = doc.doc_text[0].replace("&atilde;", u"ã")
		doc.doc_text[0] = doc.doc_text[0].replace("&hyph;", u"-")
		doc.doc_text[0] = doc.doc_text[0].replace("&eacute", u"é")
		doc.doc_text[0] = doc.doc_text[0].replace("&ntilde;", u"ñ")
		doc.doc_text[0] = doc.doc_text[0].replace("&agrave;", u"à")
		doc.doc_text[0] = doc.doc_text[0].replace("&oacute;", u"ó")
		doc.doc_text[0] = doc.doc_text[0].replace("&egrave;", u"è")
		doc.doc_text[0] = doc.doc_text[0].replace("&ocirc;", u"ô")
		doc.doc_text[0] = doc.doc_text[0].replace("&aacute;", u"á")
		doc.doc_text[0] = doc.doc_text[0].replace("&uuml;", u"ü")
		doc.doc_text[0] = doc.doc_text[0].replace("&rsquo;", u"'")
		doc.doc_text[0] = doc.doc_text[0].replace("&iacute;", u"í")
		doc.doc_text[0] = doc.doc_text[0].replace("&cent;", u"¢")
		doc.doc_text[0] = doc.doc_text[0].replace("&ccedil;", u"ç")

		#remove other ampersand strings
		ampersand_char = re.findall("&\w+;", doc.doc_text[0])
		ampersand_set = set(ampersand_char)

		for item in ampersand_set:
			doc.doc_text[0] = doc.doc_text[0].replace(item, " ")
		
		# #check IP address
		ip_address = re.findall(r"\b(?:\d{1,3}\.){3}\d{1,3}\s", doc.doc_text[0])

		if ip_address != []:
			
			for item in ip_address:

				doc.doc_text[0] = doc.doc_text[0].replace(item, "<")


		# #check URL

		urls = re.findall(r'(http|ftp|https)://([\w_-]+(?:(?:\.[\w_-]+)+))([\w.,@?^=%&:/~+#-]*[\w@?^=%&/~+#-])?', doc.doc_text[0])

		if urls != []:

			for item in urls:
				doc.doc_text[0] = doc.doc_text[0].replace(item, "<")


		# #check for email

		emails = re.findall(r'\b[a-z0-9\._%+-]+@[a-z0-9.-]+\.[a-z]{2,4}\b', doc.doc_text[0])
		
		if emails != []:

			for item in emails:
				doc.doc_text[0] = doc.doc_text[0].replace(item, "<")

		# #standardize digit format

		numerals = re.findall(r'((?:\d+[,\.])+\d+)\b', doc.doc_text[0])
		#print numerals

		if numerals != []:

			for item in numerals:

					doc.doc_text[0] = doc.doc_text[0].replace(item, "<")


		# #check for monetary symbols with numbers

		money = re.findall(r'(?:[$€¢£](?:\d+[,\.])+\d+)', doc.doc_text[0])

		if money != []:

			for cash in money:

				doc.doc_text[0] = doc.doc_text[0].replace(cash, "<")

		# #deal with abbreviations

		abbs = re.findall(r"(?:[a-z]\.){2,100}", doc.doc_text[0])
		if abbs != []:

			for thing in abbs:
				doc.doc_text[0] = doc.doc_text[0].replace(thing, "<")

		#remove hyphenations

		hyphs = re.findall(r"\b([\d\w]+-[\d\w]+)\b", doc.doc_text[0])

		if hyphs != []:

			for il in hyphs:
				doc.doc_text[0] = doc.doc_text[0].replace(il, "<")

		#remove number

		nums = re.findall(r"\d+", doc.doc_text[0])

		if nums != []:

			for yu in nums:
				doc.doc_text[0] = doc.doc_text[0].replace(yu, "<")

		# #find all dates in doc_text
		mmddyyyy_yy= re.findall("(\d+[-\\/]\d+[-\\/]\d+)\D", doc.doc_text[0])

		for d in mmddyyyy_yy:

			doc.doc_text[0] = doc.doc_text[0].replace(d, "<")

		mmmddyyyy = re.findall("((?:jan|feb|mar|apr|jun|jul|aug|sept|oct|nov|dec)[,\\.\s-]+\d+[\s,-]+\d+)\D", doc.doc_text[0])

		for d in mmmddyyyy:

			doc.doc_text[0] = doc.doc_text[0].replace(d, "<")

		nameddyyyy = re.findall("((?:january|february|march|april|may|june|july|august|september|october|november|december)\s+\d+(?:,|\s)+\d+)\D", doc.doc_text[0])

		for d in nameddyyyy:

			doc.doc_text[0] = doc.doc_text[0].replace(d, "<")

		ddnameyyyy = re.findall("\D(\d{1,2}[-\s]+(?:january|february|march|april|may|june|july|august|september|october|november|december|jan|feb|mar|apr|jun|jul|aug|sept|oct|nov|dec)[-\\.,\s]+\d+)[^0-9]", doc.doc_text[0])

		for d in ddnameyyyy:

			doc.doc_text[0] = doc.doc_text[0].replace(d, "<")
		
		return doc

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

	def tokenize(self, word_input):

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

	def ingest(self, doc):

		self.index.current_doc = doc.doc_id
		
		#make a phrase tokenizer
		if self.index_type == 'phrase_index':

			tokens = self.find_phrases(doc.doc_text[0])

		else:
			words = doc.doc_text[0].split(" ")
			#remove blanks
			words = [x for x in words if x != ""]

			tokens = map(self.tokenize, words)

			tokens = [item for sublist in tokens for item in sublist]

		self.take_tokens(tokens)


	def build_inverted_index_small_buffer_count(self):

		cwd = os.getcwd()

		ii_output = open(self.index_type+"_"+self.time_stamp+".txt", "w")

		merge_file_count = 0
		tuple_heap = []
		h.heapify(tuple_heap)

		current_term_id = 0
		max_term_id = self.index.lexicon_count-1

		while self.temp_file_count > MERGE_FILE_SIZE:

			new_temp_files = []

			#number of merge files to make
			if self.temp_file_count % MERGE_FILE_SIZE == 0:
				merges = self.temp_file_count/MERGE_FILE_SIZE
			else:
				merges = self.temp_file_count/MERGE_FILE_SIZE + 1

			temp_files_per_merge = MERGE_FILE_SIZE

			last_line = {}

			while len(new_temp_files) < merges:

				merge_file_name = "merge"+str(merge_file_count)
				print merge_file_name
				merge_file_count+=1
				merge_file = open(cwd + "/temp"+self.time_stamp+"/"+merge_file_name, "w")

				file_bufs = []

				first_file = len(new_temp_files) * temp_files_per_merge
				end_file = len(new_temp_files) * temp_files_per_merge + temp_files_per_merge

				new_temp_files.append(merge_file_name)

				for i in range(first_file, end_file):

					if i < self.temp_file_count:
						f = open(cwd + "/temp"+self.time_stamp+"/"+self.temp_file_names[i])
						print self.temp_file_names[i]
						file_bufs.append(f)
						last_line.update({f.name:""})

				current_term_id = 0
				while current_term_id <= max_term_id:

					# print current_term_id
					# raw_input()

					next = ""

					for buf in file_bufs:

						break_loop = False

						while break_loop == False:

							if last_line[buf.name] != "":

								next = last_line[buf.name]
								last_line[buf.name] = ""

							else:

								next = buf.next()

							if next == "end":

								last_line[buf.name] = "end"
								break_loop = True

							else:

								next_split = next.split(" ")

								if int(next_split[0]) == current_term_id:

								
									h.heappush(tuple_heap , next)

								else:

									last_line[buf.name] = next
									break_loop = True

					while tuple_heap != []:

						merge_file.write(h.heappop(tuple_heap))

					current_term_id += 1
				
				merge_file.write("end")
				merge_file.close()


			self.temp_file_names = new_temp_files
			self.temp_file_count = len(new_temp_files)

		#now there are less than MERGE_FILE_SIZE files

		#now merge them all into ii

		file_bufs = []
		last_line = {}

		current_term_id2 = 0
		max_term_id2 = self.index.lexicon_count -1

		for i in self.temp_file_names:

			f = open(cwd + "/temp"+self.time_stamp+"/"+i)
			file_bufs.append(f)
			last_line.update({f.name:""})

		while current_term_id2 <= max_term_id2:

			next = ""

			for buf in file_bufs:

				break_loop = False
				while break_loop == False:

					if last_line[buf.name] != "":

						next = last_line[buf.name]
						last_line[buf.name] = ""

					else:

						next = buf.next().strip("\n")

					if next == "end":

						last_line[buf.name] = "end"
						break_loop = True

					else:

						next_split = next.split(" ")

						if int(next_split[0]) == current_term_id2:

						
							if self.index_type != 'single_term_positional_index':
								
								h.heappush(tuple_heap, (next_split[1], next_split[2]))

							else:

								h.heappush(tuple_heap, (next_split[1], next_split[2], tuple(next_split[3:])))


						else:

							last_line[buf.name] = next
							break_loop = True

			ii_output.write(str(current_term_id2) + " " + str(len(tuple_heap)))

			while tuple_heap != []:

				to_write = h.heappop(tuple_heap)
				ii_output.write(" " +to_write[0] +","+ to_write[1])

				if self.index_type == 'single_term_positional_index':
					ii_output.write(",{")
					for item in to_write[2]:
						ii_output.write("_" + str(item))
					ii_output.write("}")

			ii_output.write("\n")

			current_term_id2 += 1
		
			
		ii_output.close()				
		
	def write_index_to_disk(self):

		ii = self.index.inverted_index

		
		wfile = open(self.index_type+"_" + self.time_stamp + '.txt', 'w')

		for key in ii.keys():
			wfile.write(str(key) +" " + str(len(ii[key])))
			for tupl in ii[key]:
				wfile.write(" " + str(tupl[0]) + "," + str(tupl[1]))

				if self.index_type == 'single_term_positional_index':
					wfile.write(",{")
					for item in tupl[2]:
						wfile.write("_" + str(item))
					wfile.write("}")

			wfile.write("\n")


	def make_write_temp_file(self):

		cwd = os.getcwd()
		temp_file_name = "temp" + str(self.temp_file_count)
		self.temp_file_names.append(temp_file_name)
		self.temp_file_count += 1

		#sort the list of triples

		#how to sort a list of tuple

		#sort by term id and doc id

		self.index.posting_list = sorted(self.index.posting_list, key=lambda element: (element[0], element[1]))

		if os.path.exists(cwd+"/temp"+self.time_stamp+"/") == False:

			os.mkdir(cwd+"/temp"+self.time_stamp+"/")

		temp = open(cwd+ "/temp"+self.time_stamp+"/"+temp_file_name, 'wb')
		print "making " + temp_file_name

		if self.index_type != "single_term_positional_index":

			for tupl in self.index.posting_list:

				temp.write(str(tupl[0])+ " "+ str(tupl[1]) + " " + str(tupl[2]) + "\n")

		else:

			for tupl in self.index.posting_list:

				for item in tupl:
					temp.write(str(item) + " ")
				temp.write("\n")


		temp.write("end")

		del self.index.posting_list[:]
		self.index.posting_list_count = 0


	def run(self):

		start_time = time.time()

		time_seconds =time.time()
		time_stamp = datetime.datetime.fromtimestamp(time_seconds).strftime('%Y-%m-%d_%H:%M:%S')
		self.time_stamp = time_stamp

		stop_words_file = open(STOP_WORDS_PATH, 'r')
		for stop in stop_words_file:
			self.stop_words.append(stop.strip('\n'))

		f = open(DOC_NAMES_FILE, 'r')
		for name in f:

			print("processing document: " + name.strip("\n"))

			doc_tag_count = 0

			doc = open( DOC_FILES_PATH + name.strip("\n"), 'r')

			for line in doc:
						
				if "<DOC>" in line:
							
					doc_tag_count += 1
					text_tag_count = 0
					current_doc = None

				elif "</DOC>" in line:
					doc_tag_count += 1

					if self.index_type == 'phrase_index':

						current_doc = self.normalize_phrase(current_doc)
					
					else:

						current_doc = self.normalize(current_doc)

					self.ingest(current_doc)

					if self.memory_restriction != 'unlimited' and self.index.posting_list_count >= int(self.memory_restriction):

						self.make_write_temp_file()
				

				elif doc_tag_count%2 != 0:

					if "<DOCNO>" in line:

						strip1 = line.strip("<DOCNO> ")
						strip2 = strip1.strip(" </DOCNO>\r\n")

						current_doc = self.Doc(strip2, name)
							
						self.doc_list.append(current_doc) 

					elif "<!-" not in line:
							
						if "<TEXT>" in line or "</TEXT>" in line:
							
							text_tag_count +=1
							
						elif text_tag_count %2 != 0:

							text = line.strip("\r\n")

							if text != "":
								if text != " ":
							
									current_doc.add_text(text)

		#set rest of posting list tuples to tmp file
		if self.index.posting_list_count > 0 and self.memory_restriction != 'unlimited':

			self.make_write_temp_file()


		up_to_merge = time.time()
		print "just made temp files, will merge now"
		print up_to_merge - start_time

		#make inverted index if temp files were made

		# print "lexicon_count " + str(self.index.lexicon_count)
		if self.memory_restriction != 'unlimited':
		
			print "building inverted index from temp files"

			self.build_inverted_index_small_buffer_count()

		else:#the inverted index is in memory and write to disk
			self.write_index_to_disk()

		merged = time.time()
		print "merged temp file"
		print merged - up_to_merge


def setup(arg_list):

	command_line_index_type = ['single_term_index', 'single_term_positional_index', 'stem_index', 'phrase_index']
	command_line_memreq = ['1000', '10000', '100000', 'unlimited']

	index_type = "-"
	memory_restriction = "-"
	im = None

	for item in arg_list:

		if item in command_line_index_type:

			index_type = item

		if item in command_line_memreq:

			memory_restriction = item

	if index_type != "-" and memory_restriction != "-":

		im = Index_Machine(index_type = index_type, memory_restriction = memory_restriction)
	
	elif index_type == "-" and memory_restriction != "-":

		im = Index_Machine(memory_restriction = memory_restriction)

	elif index_type != "-" and memory_restriction == "-":

		im = Index_Machine(index_type = index_type)

	else:

		im = Index_Machine()

	print("running index_machine.py with command line inputs: " + index_type +", "+ memory_restriction)
	print("default index_type is single_term_index and default memory_restriction is unlimited")
	
	return im

def main():
	
	#the index machine parses documents from the text files
	#does the normalization and tokenization of documents and sends them to the index
	#and controls the flow of documents into the index for memory size management

	start_time = time.time()
	
	im = setup(sys.argv)

	im.run()

	elapsed_time = time.time() - start_time

	print "total time elapsed"
	print elapsed_time


if __name__ == "__main__":
	main()			

