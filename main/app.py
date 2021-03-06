# Built-in Imports
import os
import io
import sys
import jsonify
import json
import numpy as np
import pandas as pd
import pickle
from datetime import datetime
from base64 import b64encode
import base64
from io import BytesIO #Converts data from Database into bytes
from datetime import datetime
from pathlib import Path
import shutil

# Utility
# comment this when running python3 create_db.py
from .DataLoader import DataLoader
from .DataProjector import DataProjector
from .UserModelCoupled import UserModelCoupled
from .Utils import *

# Tesseract
import pytesseract
from PIL import Image

# Gensim
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel

# Flask
import flask
from flask import Flask, render_template, request, flash, redirect, url_for, send_file # Converst bytes into a file for downloads
from flask import request, Response

# FLask SQLAlchemy, Database
from flask_sqlalchemy import SQLAlchemy

basedir = 'sqlite:///' + os.path.join(os.path.abspath(os.path.dirname(__file__)), 'data.sqlite')


app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = basedir
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['SECRET_KEY'] = 'dev'
app.config["DEBUG"] = True
db = SQLAlchemy(app)



# Picture table. By default the table name is filecontent
class FileContent(db.Model):

    """ 
    The first time the app runs you need to create the table. In Python
    terminal import db, Then run db.create_all()
    """
    """ ___tablename__ = 'yourchoice' """ # You can override the default table name

    id = db.Column(db.Integer,  primary_key=True)
    name = db.Column(db.String(128), nullable=False)
    data = db.Column(db.LargeBinary, nullable=False) 
    rendered_data = db.Column(db.Text, nullable=False)
    userid = db.Column(db.String(64))
    text = db.Column(db.Text)
    entities = db.Column(db.Text)
    location = db.Column(db.String(64))
    pic_date = db.Column(db.DateTime, nullable=False)
    def __repr__(self):
        return f'Pic Name: {self.name} Data: {self.data} text: {self.text} created on: {self.pic_date} location: {self.location}'


# Picture table. By default the table name is filecontent
class RecContent(db.Model):

    """ 
    The first time the app runs you need to create the table. In Python
    terminal import db, Then run db.create_all()
    """
    """ ___tablename__ = 'yourchoice' """ # You can override the default table name

    id = db.Column(db.Integer,  primary_key=True)
    userid = db.Column(db.String(64))
    text = db.Column(db.Text)
    rec_date = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)
    def __repr__(self):
        return f'User: {self.userid} created on: {self.rec_date} text: {self.text}'

# Index
@app.route('/index', methods=['GET', 'POST'])
@app.route('/')
def index():

    pics = FileContent.query.all()
    if pics: # This is because when you first run the app, if no pics in the db it will give you an error
        all_pics = pics
        if request.method == 'POST':

            flash('Upload succesful!')
            return redirect(url_for('upload'))  

        return render_template('index.html', all_pic=all_pics)
    else:
        return render_template('index.html')

# Query
@app.route('/query')
def query():

    all_pics = FileContent.query.all()
    return render_template('query.html', all_pic=all_pics)

# Corpus
@app.route('/corpus')
def corpus():
    query = FileContent.query.with_entities(FileContent.userid.distinct())
    all_uids = [i[0] for i in query.all()]
    return render_template('corpus.html', all_uids=all_uids)

# Render the pics
def render_picture(data):
    
    render_pic = base64.b64encode(data).decode('ascii') 
    return render_pic

# Upload
@app.route('/upload', methods=['POST'])
def upload():

    file = request.files['inputFile']
    data = file.read()
    render_file = render_picture(data)
    text = request.form['text']
    location = request.form['location']
    userid = ''

    newFile = FileContent(name=file.filename, data=data, rendered_data=render_file, text=text, location=location, userid=userid, pic_date=datetime.utcnow, entities='{}')
    db.session.add(newFile)
    db.session.commit() 
    full_name = newFile.name
    full_name = full_name.split('.')
    file_name = full_name[0]
    file_type = full_name[1]
    file_date = newFile.pic_date
    file_location = newFile.location
    file_render = newFile.rendered_data
    file_id = newFile.id
    file_text = newFile.text

    return render_template('upload.html', file_name=file_name, file_type=file_type, file_date=file_date, file_location=file_location, file_render=file_render, file_id=file_id, file_text=file_text)


@app.route('/upload.php', methods=['POST'])
def upload_php():

    file = request.files['image']
    data = file.read()
    render_file = render_picture(data)
    lang = request.form['lang']
    # convert screen to text - tesseract
    text = pytesseract.image_to_string(Image.open(file.stream), lang=lang)
    # ibm nlp
    entities = detect_entities(text)

    location = request.form['extra']
    userid = request.form['username']
    pic_date = filenameToTime(json.loads(request.form['extra'])['filename'])

    newFile = FileContent(name=file.filename.split('/')[-1], data=data, rendered_data=render_file, text=text, location=location, userid=userid, pic_date=pic_date, entities=entities)
    db.session.add(newFile)
    db.session.commit() 

    return "file uploaded"


# Download
@app.route('/download/<int:pic_id>')
def download(pic_id):

    file_data = FileContent.query.filter_by(id=pic_id).first()
    file_name = file_data.name
    return send_file(BytesIO(file_data.data), attachment_filename=file_name, as_attachment=True)

# Download
@app.route('/build/<string:user_id>')
def build(user_id):
    model_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'models/'+user_id)
    Path(model_path).mkdir(parents=True, exist_ok=True)
    file_data = FileContent.query.filter((FileContent.userid == user_id))
    screens = [screen.text for screen in file_data]
    entities = [getEntities(screen.entities) for screen in file_data]
    apps = [getApp(screen.location) for screen in file_data]
    docs = [getDoc(screen.location) for screen in file_data]
    buildCorpus(model_path,screens,entities,apps,docs)

    #create matrix for retrieval
    # Path(os.path.join(model_path,'temp')).rmdir()
    if os.path.exists(os.path.join(model_path,'temp')):
        shutil.rmtree(os.path.join(model_path,'temp'))
    data = DataLoader(model_path)
    params["corpus_directory"] = model_path
    projector = DataProjector(data, params, model_path)
    projector.generate_latent_space()
    projector.create_feature_matrices()

    query = FileContent.query.with_entities(FileContent.userid.distinct())
    all_uids = [i[0] for i in query.all()]

    return render_template('corpus.html', all_uids=all_uids)


# Show Pic
@app.route('/pic/<int:pic_id>')
def pic(pic_id):

    get_pic = FileContent.query.filter_by(id=pic_id).first()

    return render_template('pic.html', pic=get_pic)

# Update
@app.route('/update/<int:pic_id>', methods=['GET', 'POST'])
def update(pic_id):

    pic = FileContent.query.get(pic_id)

    if request.method == 'POST':
        pic.location = request.form['location']
        pic.text = request.form['text']

        db.session.commit()
        flash(f'{pic.name} Has been updated')
        return redirect(url_for('index'))
    return render_template('update.html', pic=pic)


#Delete
@app.route('/<int:pic_id>/delete', methods=['GET', 'POST'])
def delete(pic_id):

    del_pic = FileContent.query.get(pic_id)
    if request.method == 'POST':
        form = request.form['delete']
        if form == 'Delete':
            print(del_pic.name)
            db.session.delete(del_pic)
            db.session.commit()
            flash('Picture deleted from Database')
            return redirect(url_for('index'))
    return redirect(url_for('index'))


# Load LDA, regression models
# data = DataLoader()
# id2word, corpus, lda_model, recs, num_topic, max_num_recs, reg_model = data.id2word, data.corpus, data.lda_model, data.recs, data.num_topic, data.max_num_recs, data.reg_model
# num_recs = 1000,



@app.route('/laboratory.php', methods=['POST'])
def predict():

    file = request.files['image']
    data = file.read()
    render_file = render_picture(data)
    lang = request.form['lang']
    # convert screen to text - tesseract
    text = pytesseract.image_to_string(Image.open(file.stream), lang=lang)
    # ibm nlp
    entities = detect_entities(text)

    location = request.form['extra']
    userid = request.form['username']
    pic_date = filenameToTime(json.loads(request.form['extra'])['filename'])

    newFile = FileContent(name=file.filename.split('/')[-1], data=data, rendered_data=render_file, text=text, location=location, userid=userid, pic_date=pic_date, entities=entities)
    db.session.add(newFile)
    db.session.commit() 

    # Predict here
    # userid = 'C1MR3058G940'
    model_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'models/'+userid)
    data = DataLoader(model_path)
    # data.print_info()
    params["corpus_directory"] = model_path
    projector = DataProjector(data, params, model_path)
    projector.generate_latent_space()
    projector.create_feature_matrices()

    for method_ind in range(num_methods):
        method = Method_list[method_ind]

        selected_terms = []        # ID of terms that the user has given feedback to
        feedback_terms = []        # feedback value on the selected terms
        recommended_terms = []     # list of ID of terms that have been recommended to the user
        selected_docs = []         # ID of snapshots that the user has given feedback to (may not be available in practice)
        feedback_docs = []         # feedback value on the selected snapshots (may not be available in practice)

        print('Loading real-time generated snapshots...')
        all_online_docs = []   # all snapshots generated from realtime user activity
        fv_online_docs = []    # considered snapshots generated from realtime user activity
        fb_online_docs = []    # dummy feedback for the newly generated snapshots

        online_data = FileContent.query.filter((FileContent.userid == userid))
        # print(online_data[-1])
        online_screens = [screen.text for screen in [online_data[-1]]]
        online_entities = [getEntities(screen.entities) for screen in [online_data[-1]]]
        online_apps = [getApp(screen.location) for screen in [online_data[-1]]]
        online_docs = [getDoc(screen.location) for screen in [online_data[-1]]]
        fv_online_docs = getOnlineDocs(model_path, online_screens, online_entities, online_apps, online_docs)
        fb_online_docs+= [1 for i in range(len(fv_online_docs))]


        if method == "LSA-coupled-Thompson":
            # initialize the user model in the projected space
            user_model = UserModelCoupled(params)
            # create the design matrices for docs and terms
            user_model.create_design_matrices(projector, selected_terms, feedback_terms,selected_docs, feedback_docs, fv_online_docs, fb_online_docs)
            #user_model.create_design_matrices(projector, selected_terms, feedback_terms, [1], [2], [[(1,2),(4,3)], [(2,2),(14,1)] ], [0.5, 0.1])
            # posterior inference
            user_model.learn()
            # Thompson sampling for coupled EVE
            #TODO: test having K thompson sampling for the K recommendations
            if params["Thompson_exploration"]:
                theta = user_model.thompson_sampling()
            else:
                theta = user_model.Mu # in case of no exploration, use the mean of the posterior
            scored_docs = np.dot(projector.doc_f_mat, theta)
            scored_terms = np.dot(projector.term_f_mat, theta)
            #print theta

        if method == "LSA-coupled-UCB":
            # initialize the user model in the projected space
            user_model = UserModelCoupled(params)
            # create the design matrices for docs and terms
            user_model.create_design_matrices(projector, selected_terms, feedback_terms,selected_docs, feedback_docs, fv_online_docs, fb_online_docs)
            # posterior inference
            user_model.learn()
            # Upper confidence bound method
            scored_docs = user_model.UCB(projector.doc_f_mat)
            scored_terms = user_model.UCB(projector.term_f_mat)

        if method == "Random":
            scored_docs = np.random.uniform(0,1,projector.num_docs)
            scored_terms = np.random.uniform(0,1,projector.num_terms)

        #---------------------- 3.4: gather user feedback ---------------------------#
        #sort items based on their index
        #todo: if time consuming then have k maxs instead of sort
        sorted_docs = sorted(range(len(scored_docs)), key=lambda k:scored_docs[k], reverse=True)
        # make sure the selected items are not recommended to user again
        sorted_docs_valid = [doc_idx for doc_idx in sorted_docs if doc_idx not in set(selected_docs)]

        # make sure the selected terms are not recommended to user again
        sorted_terms = sorted(range(len(scored_terms)), key=lambda k:scored_terms[k], reverse=True)

        sorted_views_list = []  # sorted ranked list of each view
        for view in range(1, data.num_views):
            # sort items of each view. Exclude (or not exclude) the previously recommended_terms.
            if params["repeated_recommendation"]:
                sorted_view = [term_idx for term_idx in sorted_terms
                               if term_idx not in set(selected_terms) and data.views_ind[term_idx] == view]
            else:
                sorted_view = [term_idx for term_idx in sorted_terms
                               if term_idx not in set(recommended_terms) and data.views_ind[term_idx] == view]

            sorted_views_list.append(sorted_view)

        for view in range(1, data.num_views):
            print('view %d:' %view)
            for i in range(min(params["suggestion_count"],data.num_items_per_view[view])):
                print('    %d,' %sorted_views_list[view-1][i] + ' ' + data.feature_names[sorted_views_list[view-1][i]])
        print('Relevant document IDs (for debugging):')
        for i in range(params["suggestion_count"]):
            print('    %d' %sorted_docs_valid[i])

    newRec = RecContent(userid=userid, text=str({'type':sorted_views_list}))
    db.session.add(newRec)
    db.session.commit() 
    return "file uploaded"

# Upload
@app.route('/retrieve', methods=['POST'])
def retrieve():
    userid = 'C1MR3058G940'
    all_recs = RecContent.query.filter_by(userid=userid)
    return {'recs':[rec.text for rec in [all_recs[-1]]]}

# @app.route('/upload', methods=['POST'])
# def upload():
# 	return {'type':'upload'}

if __name__ == "__main__":
	pass
