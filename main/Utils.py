import os
import pandas as pd
import numpy as np
import json
import io
import requests
import sys
import pickle
import itertools
import base64
from io import BytesIO

from collections import defaultdict
from datetime import datetime
from urllib.parse import urlparse, unquote

# Tesseract
import pytesseract
from PIL import Image, ImageChops

# Gensim
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess

# IBM Watson NLP
from ibm_watson import NaturalLanguageUnderstandingV1
from ibm_watson.natural_language_understanding_v1 import Features, EntitiesOptions, KeywordsOptions
from ibm_cloud_sdk_core.authenticators import IAMAuthenticator

# spacy for lemmatization
import spacy
import en_core_web_sm

# NLTK Stop words
import nltk
nltk.download("stopwords")
from nltk.corpus import stopwords
stop_words = stopwords.words('english')
nlp = en_core_web_sm.load()


params = {

    # Number of recommended entities from each view
    "suggestion_count": 10,
    # Number of online snapshots to consider (the latest snapshots)
    "imp_doc_to_consider": 4,
    # True: normalize TF-IDF weights to sum to 1, False: no normalization. TODO: DOES THIS MAKE SENSE?
    "normalize_terms": True,
    # True: use exploration algorithm (Thompson Sampling) for recommendation, False: use the mean of the estimate.
    "Thompson_exploration": False,
    # True: allow the algorithm to show previously recommended items, False: each item can be recommended only once
    "repeated_recommendation": True,
    # A heuristic method to shrink the variance of the posterior (reduce the exploration). it should be in (0,1];
    "exploration_rate": 1,  # NOT IMPLEMENTED YET
    # Number of iterations of the simulated study
    "num_iterations": 50,
    # Number of latent dimensions for data representation
    "num_latent_dims": 100,
    # Number of runs (only for the simulated study, set to 1 for real data setting)
    "num_runs": 1,  # NOT IMPLEMENTED YET
    # True: prepare the data for UI but have the interaction in the terminal
    "UI_simulator": True,
    # The directory of the corpus (It should have /corpus.mm, /dictionary.dict, and views_ind_1.npy files)
    #"corpus_directory": 'corpus1_2/corpus7_sim',
    "corpus_directory": 'corpus1_2/P01',
    # The directory of the new snapshots that will be checked at the beginning of each iteration
    "snapshots_directory": 'user activity',
    # True: Use the simulated user data to simulate the user feedback
    "Simulated_user": False,
}

# Set the desirable method to True for the experiment
Methods = {
    "LSA-coupled-Thompson": True,
    "LSA-coupled-UCB": False,
    "Random": False
}
Method_list = []
num_methods = 0
for key in Methods:
    if Methods[key] == True:
        Method_list.append(key)
        num_methods = num_methods + 1

def filenameToTime(text):
	return datetime.utcfromtimestamp( float(text.replace(',','.').replace('_idle','')) )
def detect_entities(mytext):
    api_keys = [
        ['510PxKJWLcqRcywFcfvTcjVfePAr03FjjifzKzXoTgAX','https://api.eu-gb.natural-language-understanding.watson.cloud.ibm.com/instances/3e06a67a-c07c-48f7-a122-441887dffe31'],
        ['B27mc3Cwbn8WM0I7KZA8bTX_zSlPI0DYSM4cA5iR0yhw','https://api.eu-gb.natural-language-understanding.watson.cloud.ibm.com/instances/db810944-aba3-4e4f-a09e-bb36fe56b939'],
        ['NQqmqzPxC9k-uuvwIufkTfP7bnRhinw07ZBM16g_Q9L5','https://api.eu-gb.natural-language-understanding.watson.cloud.ibm.com/instances/bf017fc1-0a4f-47c4-bb99-f57f1d9a85a1'],
        ['E8hWog4WEv4U5GSCWelj8-LA-uoNHR309V4Dwtkl0CGZ','https://api.eu-gb.natural-language-understanding.watson.cloud.ibm.com/instances/6368e63c-c876-4cb3-9831-b3aacf9e49a2'],
        ['0nN9UNR6KyKKVoqRXaaNrIuHhQwFoBBtGp5mhTXyzaxa','https://api.eu-gb.natural-language-understanding.watson.cloud.ibm.com/instances/3ede59cd-1df9-4a5e-9b01-e8e94e22b2e4'],
        ['BQ66VkFQAX5J99zgNvZlrQThDkgGv0VuwTETfnYWv_vL','https://api.eu-gb.natural-language-understanding.watson.cloud.ibm.com/instances/72a0210b-b5fb-43e9-9d90-48aec2ecbf98'],
        ['5PZzZznrqAG5B4o4kgs2rMBYLkfRndh_RDJnQa9qw70E','https://api.eu-gb.natural-language-understanding.watson.cloud.ibm.com/instances/ad303cf9-799e-4774-b7ad-1e26a3e78c72'],
        ['gpR4DPPZATUqvpMD7sjKhmg1TCI9KxEoYha_nJigcDti','https://api.eu-gb.natural-language-understanding.watson.cloud.ibm.com/instances/6cd6a44c-bea4-40d8-89f3-d16b2df21e08'],
        ['fh3zzAkRExKT1yVpV_kSFJztYS7yoeltXv2neUKg85Z_','https://api.eu-gb.natural-language-understanding.watson.cloud.ibm.com/instances/4a16e4a8-6812-47ab-91a2-a12101367082']
    ]
    for key in api_keys:
        apikey = key[0]
        apiurl = key[1]
        authenticator = IAMAuthenticator(apikey)
        service = NaturalLanguageUnderstandingV1(
            version='2018-03-16',
            authenticator=authenticator)
        service.set_service_url(apiurl)

        try:
            response = service.analyze(text=mytext,
                features=Features(entities=EntitiesOptions(limit=1000),
                                  keywords=KeywordsOptions(limit=1000))).get_result()
            return json.dumps(response)
            break
        except Exception as e:
            print('Failed to request: '+ str(e))
            if 'unsupported text language' in str(e) or 'unknown language detected' in str(e) or 'Code: 400' in str(e):
                break
    return '{}'

if __name__ == '__main__':
	testdata = {'control': 'Complete'}
	print(preprocess(testdata))
def iter_docs(screens, entities, apps, docs, webqueries, stoplist, amount_docs_already_index):
    for idx, text in enumerate(screens):
        if (idx >= amount_docs_already_index):
            text = text.strip().lower()
            entities_in_text = []
            for e in entities[idx]:
            	raw_e = e.replace('_',' ')
            	for i in range(0, text.count(raw_e)):
            		entities_in_text.append(e)
            	text = text.replace(raw_e, "")
            texts = [x for x in
                    gensim.utils.tokenize(text, lowercase=True, deacc=True,
                                         errors="ignore")
                   if x not in stoplist and len(x)>3]
            texts+= entities_in_text
            texts+= [apps[idx]]
            texts+= [docs[idx]]
            texts+= [webqueries[idx]]
            yield (x for x in texts)
class MyCorpus(object):
    def __init__(self, screens, entities, apps, docs, webqueries, stoplist, amount_docs_already_index):
        self.screens = screens
        self.entities = entities
        self.stoplist = stoplist
        self.amount_docs_already_index = amount_docs_already_index
        self.texts = iter_docs(screens, entities, apps, docs, webqueries, stoplist, amount_docs_already_index)
        texts_for_frequency = iter_docs(screens, entities, apps, docs, webqueries, stoplist, amount_docs_already_index)
        frequency = defaultdict(int)
        for text in texts_for_frequency:
            for token in text:
                frequency[token] += 1
        self.texts = [[token for token in text if frequency[token] > 0]
                      for text in self.texts]
        self.dictionary = gensim.corpora.Dictionary(self.texts)
        self.size = len(screens)
    def __iter__(self):
        for text in self.texts:
            yield self.dictionary.doc2bow(text)
def buildCorpus(model_path, screens, entities, apps, docs, webqueries):
	#check if model was saved and load lsi model
	stoplist = set(nltk.corpus.stopwords.words("english"))
	corpus_already_index = None
	corpus_path = os.path.join(model_path,'corpus.mm')
	merged_corpus_path = os.path.join(model_path,'merged_corpus.mm')
	dict_path = os.path.join(model_path,'dictionary.dict')

	if (os.path.isfile(corpus_path)):
		corpus_already_index = corpora.MmCorpus(corpus_path)
		corpus_already_index.dictionary = gensim.corpora.Dictionary.load(dict_path, mmap=None)
		amount_docs_already_index = len(corpus_already_index)

	# if saved model exist
	if corpus_already_index != None:
		if (len(screens)>amount_docs_already_index):
			corpus = MyCorpus(screens, entities, apps, docs, webqueries, stoplist, amount_docs_already_index)
			dict2_to_dict1 = corpus_already_index.dictionary.merge_with(corpus.dictionary)
			merged_corpus = itertools.chain(corpus_already_index,dict2_to_dict1[corpus])
			corpora.MmCorpus.serialize(merged_corpus_path,merged_corpus)
			corpus=corpora.MmCorpus(merged_corpus_path)
			corpus.dictionary = corpus_already_index.dictionary
			corpora.MmCorpus.serialize(corpus_path, corpus)
			corpus.dictionary.save(dict_path)
		else:
			corpus = corpus_already_index
	else:
		corpus = MyCorpus(screens, entities, apps, docs, webqueries, stoplist, 0)
		corpora.MmCorpus.serialize(corpus_path, corpus)
		corpus.dictionary.save(dict_path)

	# building views file
	corpus = corpora.MmCorpus(corpus_path)
	corpus.dictionary = gensim.corpora.Dictionary.load(dict_path, mmap=None)
	dictionary_view = {}
	entity_view = []
	# print(entities)
	for e in entities:
		entity_view+= e
	for entity_id in corpus.dictionary.doc2bow(entity_view):
		dictionary_view[entity_id[0]] = 1
	for app_id in corpus.dictionary.doc2bow(apps):
		dictionary_view[app_id[0]] = 2
	for doc_id in corpus.dictionary.doc2bow(docs):
		dictionary_view[doc_id[0]] = 3
	for webquery_id in corpus.dictionary.doc2bow(webqueries):
		dictionary_view[webquery_id[0]] = 4
	feature_names = []
	for i in range(corpus.num_terms):
		if i in dictionary_view:
			feature_names.append(dictionary_view[i])
		else:
			feature_names.append(0)
	# print(feature_names)
	np.save(os.path.join(model_path,'views_ind_1.npy'),feature_names)
def getOnlineDocs(model_path, screens, entities, apps, docs, webqueries):
	dict_path = os.path.join(model_path,'dictionary.dict')
	dictionary = gensim.corpora.Dictionary.load(dict_path, mmap=None)
	stoplist = set(nltk.corpus.stopwords.words("english"))
	all_online_docs = []
	for idx, text in enumerate(screens):

		text = text.strip().lower()
		entities_in_text = []
		for e in entities[idx]:
			raw_e = e.replace('_',' ')
			for i in range(0, text.count(raw_e)):
				entities_in_text.append(e)
			text = text.replace(raw_e, "")
		doc = [x for x in gensim.utils.tokenize(text.strip().lower(), lowercase=True, deacc=True, errors="ignore")
					if x not in stoplist and len(x)>2] + entities[idx] + [apps[idx]] + [docs[idx]] + [webqueries[idx]]
		all_online_docs+= [dictionary.doc2bow(doc)]
	return all_online_docs


def getApp(extra_info):
	detail = json.loads(extra_info)
	app = detail["appname"].lower().replace(' ','_')
	if ("safari" in app) or ("chrome" in app) or ("firefox" in app) or ("opera" in app):
		if 'file:/' not in detail["url"]:
			app = urlparse(detail["url"]).netloc.replace(".","_")
		else:
			app = 'file:::'
	return app
def getDoc(extra_info):
	detail = json.loads(extra_info)
	title = detail["title"].lower().replace(' ','_').replace('.','_')
	return title
def getWebQuery(extra_info):
	detail = json.loads(extra_info)
	url = detail["url"].lower()
	query = ''
	if ('google' in url or 'bing' in url or 'duckduckgo' in url) and 'q=' in url:
		query = unquote(url).split('q=')[1].split('&')[0].replace('+','_')
	return query
def getEntities(watson):
	if watson=='':
		return []
	detail = json.loads(watson)
	keywords = []
	if 'keywords' in detail:
		for keyword in detail['keywords']:
			# keywords+= [keyword['text'].lower().replace(' ','_')] if len(keyword['text'])>3 and (len(keyword['text'])==4 and ' ' not in keyword['text']) else []
			keywords+= [keyword['text'].lower().replace(' ','_')] if len(keyword['text'])>3 else []
	if 'entities' in detail:
		for keyword in detail['entities']:
			keywords+= [keyword['text'].lower().replace(' ','_')] if len(keyword['text'])>3 else []
	return keywords

# def convertToText(change, lang):
# 	detectText = pytesseract.image_to_string(change, lang=lang)
# 	print(detectText)
# 	return detectText

def convertToText(change, lang):
	target_url = "https://vision.googleapis.com/v1/images:annotate?key=AIzaSyDXBo-XUHlHHTN4xefvO9DmLZzCSHLhLCM"
	buffered = BytesIO()
	change.save(buffered, format="JPEG")
	encoded_string = base64.b64encode(buffered.getvalue()).decode("utf8")
	#encoded_string = base64.b64encode(change.read()).decode("utf8")

	request_data = {"requests":[{"features": [{"type": "TEXT_DETECTION"}], "image": {"content": encoded_string}}]}
	r = requests.post(target_url, data=json.dumps(request_data))
	print(r.status_code, r.reason)

	response = json.loads(r.text.encode("utf-8").strip())

	detectText = ''
	if 'responses' in response and'textAnnotations' in response['responses'][0]:
		texts = response['responses'][0]['textAnnotations']
		detectText = texts[0]['description']
		locale = texts[0]['locale']
		# writeToFile(json.dumps(response), outPath.replace("converted","google"))
		# if locale!="en" and locale!="fr" and locale!="de" and locale!="it" and locale!="ja" and locale!="ko":
		#     detect_entities_ibm(translate_text(detectText.encode('utf8').strip(), outPath, locale), outPath)
		# else:
		#     print("2) TRANSLATION IGNORED! THIS IS ENGLISH!")
		#     detect_entities_ibm(detectText, outPath)
	print(detectText)
	return detectText
