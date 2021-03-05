import os
import pandas as pd
import numpy as np
import json
import io
import sys
import pickle
import itertools

from collections import defaultdict
from datetime import datetime
from urllib.parse import urlparse

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

convertValtoInt = {
	'gender': 	{ 
					'Female': 0, 
					'Male': 1
	},
	'control': 	{ 
					'Quite a lot': 4,
					'Complete': 3,
					'A fair amount': 2,
					'Not very much': 1,
					'None': 0,
	},
	'satisfaction_General': {
					'Entirely satisfied': 4,
					'Satisfied': 3,
					'Neither satisfied nor dissatisfied': 2,
					'Dissatisfied': 1,
					'Entirely dissatisfied': 0,
	},
	'satisfaction_Sleep': 	{ 
					'Entirely satisfied': 4,
					'Satisfied': 3,
					'Neither satisfied nor dissatisfied': 2,
					'Dissatisfied': 1,
					'Entirely dissatisfied': 0,
	},
	'satisfaction_SleepEnvironment': 	{ 
					'Very suitable': 4,
					'Suitable': 3,
					'Somewhat suitable': 2,
					'Not suitable': 1,
					'Not suitable at all': 0,
	},
	'satisfaction_Diet': 	{ 
					'Very suitable': 4,
					'Suitable': 3,
					'Somewhat suitable': 2,
					'Not suitable': 1,
					'Not suitable at all': 0,
	},
	'satisfaction_Exercise': 	{ 
					'Very suitable': 4,
					'Suitable': 3,
					'Somewhat suitable': 2,
					'Not suitable': 1,
					'Not suitable at all': 0,
	}

}

AllFeatures = ["gender", "workHours", "control", "satisfaction_General", "satisfaction_Sleep", "satisfaction_SleepEnvironment", "satisfaction_Diet", "satisfaction_Exercise"]

AllRecs = {
	'sl01': "Have a short sleep before your first night shift.",
	'sl02': "If coming off night shifts, have a short sleep and go to bed earlier that night.",
	'sl03': "Once you have identified a suitable sleep schedule try to keep to it.",
	'sl04': "Sleep in your bedroom and avoid using it for other activities such as watching television, eating and working.",
	'sl05': "Use heavy curtains, blackout blinds or eye shades to darken the bedroom.",
	'sl06': "Disconnect the phone or use an answer machine and turn the ringer down.",
	'sl07': "Ask your family not to disturb you and to keep the noise down when you are sleeping.",
	'sl08': "If it is too noisy to sleep consider using earplugs, white noise or background music to mask external noises.",
	'sl09': "Adjust the bedroom temperature to a comfortable level, cool conditions improve sleep.",
	'sl10': "Avoid the use of alcohol to help you fall asleep.",
	'sl11': "Avoid the regular use of sleeping pills and other sedatives to aid sleep. These are not recommended because they can lead to dependency and addiction.",
	'sl12': "Go for a short walk, relax with a book, listen to music and/or take a hot bath before going to bed.",
	'sl13': "Avoid vigorous exercise before sleep as it is stimulating and raises the body temperature.",
	'sl14': "Don’t go to bed feeling hungry: have a light meal or snack before sleeping but avoid fatty, spicy and/or heavy meals, as these are more difficult to digest and can disturb sleep.",
	'sl15': "Plan your domestic duties around your shift schedule and try to ensure that you do not complete them at the cost of rest/sleep.",
	'al01': "Take moderate exercise before starting work which may increase your alertness during the shift.",
	'al02': "Keep the light bright at work to increase your alertness.",
	'al03': "Get up and walk around during breaks.",
	'al04': "Plan to do more stimulating work at the times you feel most drowsy.",
	'al05': "Keep in contact with co-workers as this may help both you and them stay alert.",
	'al06': "Avoid driving for long periods or a long distance after a period of night shifts or long working hours.",
	'al07': "Consider using public transport or sharing a lift with a co-worker and take it in turns to drive.",
	'so01': "Talk to friends and family about shiftwork. If they understand the problems you are facing it will be easier for them to be supportive and considerate.",
	'so02': "Make your family and friends aware of your shift schedule so they can include you when planning social activities.",
	'so03': "Make the most of your time off and plan mealtimes, weekends and evenings together.",
	'so04': "Invite others who work similar shifts to join you in social activities when others are at work and there are fewer crowds."
}

def preprocess(data):
	res = dict(data)
	for fea, val in res.items():
		if fea not in AllFeatures:
			continue
		if fea == 'workHours':
			if res[fea]<=25:
				res[fea] = 1
			if res[fea]>25 and res[fea]<=40:
				res[fea] = 2
			if res[fea]>40 and res[fea]<=59:
				res[fea] = 3
			if res[fea]>=60:
				res[fea] = 4
		else:
			res[fea] = convertValtoInt[fea][val]
	return res

# Tokenize words and Clean-up text
def sent_to_words(documents):
	for doc in documents:
		yield(gensim.utils.simple_preprocess(str(doc), deacc=True))  # deacc=True removes punctuations
# Remove Stopwords
def remove_stopwords(texts):
	return [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]
# Lemmatize
def lemmatization(texts, allowed_postags=['NOUN', 'ADJ', 'ADV']):
	"""https://spacy.io/api/annotation"""
	texts_out = []
	for sent in texts:
		doc = nlp(" ".join(sent))
		texts_out.append([token.lemma_ for token in doc])
	return texts_out
def takeSecond(elem):
	return elem[1]
def flatRecom(seq):
	seen_bl = set()
	seen_add_bl = seen_bl.add
	return [x for x in seq if not (x[0] in seen_bl or seen_add_bl(x[0]))]
def getNumRecs(max_num_recs):
	try:
		num_recs = int(sys.argv[2]) if int(sys.argv[2])<=max_num_recs else max_num_recs
		return num_recs
	except:
		return 5


params = {

    # Number of recommended entities from each view
    "suggestion_count": 5,
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
        ['510PxKJWLcqRcywFcfvTcjVfePAr03FjjifzKzXoTgAX','https://api.eu-gb.natural-language-understanding.watson.cloud.ibm.com/instances/3e06a67a-c07c-48f7-a122-441887dffe31']
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
    return '{}'

if __name__ == '__main__':
	testdata = {'control': 'Complete'}
	print(preprocess(testdata))
def iter_docs(screens, entities, apps, docs, stoplist, amount_docs_already_index):
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
            yield (x for x in texts)
class MyCorpus(object):
    def __init__(self, screens, entities, apps, docs, stoplist, amount_docs_already_index):
        self.screens = screens
        self.entities = entities
        self.stoplist = stoplist
        self.amount_docs_already_index = amount_docs_already_index
        self.texts = iter_docs(screens, entities, apps, docs, stoplist, amount_docs_already_index)
        texts_for_frequency = iter_docs(screens, entities, apps, docs, stoplist, amount_docs_already_index)
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
def buildCorpus(model_path, screens, entities, apps, docs):
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
			corpus = MyCorpus(screens, entities, apps, docs, stoplist, amount_docs_already_index)
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
		corpus = MyCorpus(screens, entities, apps, docs, stoplist, 0)
		corpora.MmCorpus.serialize(corpus_path, corpus)
		corpus.dictionary.save(dict_path)

	# building views file
	corpus = corpora.MmCorpus(corpus_path)
	corpus.dictionary = gensim.corpora.Dictionary.load(dict_path, mmap=None)
	dictionary_view = {}
	entity_view = []
	for e in entities:
		entity_view+= e
	for entity_id in corpus.dictionary.doc2bow(entity_view):
		dictionary_view[entity_id[0]] = 1
	for app_id in corpus.dictionary.doc2bow(apps):
		dictionary_view[app_id[0]] = 2
	for doc_id in corpus.dictionary.doc2bow(docs):
		dictionary_view[doc_id[0]] = 3
	feature_names = []
	for i in range(corpus.num_terms):
		if i in dictionary_view:
			feature_names.append(dictionary_view[i])
		else:
			feature_names.append(0)
	np.save(os.path.join(model_path,'views_ind_1.npy'),feature_names)
def getOnlineDocs(model_path, screens, entities, apps, docs):
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
					if x not in stoplist and len(x)>2] + entities[idx] + [apps[idx]] + [docs[idx]]
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
def getEntities(watson):
	detail = json.loads(watson)
	keywords = []
	if 'keywords' in detail:
		for keyword in detail['keywords']:
			keywords+= [keyword['text'].lower().replace(' ','_')] if len(keyword['text'])>3 and (len(keyword['text'])==4 and ' ' not in keyword['text']) else []
	if 'entities' in detail:
		for keyword in detail['entities']:
			keywords+= [keyword['text'].lower().replace(' ','_')] if len(keyword['text'])>3 and (len(keyword['text'])==4 and ' ' not in keyword['text']) else []
	return keywords