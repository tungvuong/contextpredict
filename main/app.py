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
# from .DataLoader import DataLoader
# from .DataProjector import DataProjector
# from .UserModelCoupled import UserModelCoupled
# from .Utils import *

# Tesseract
import pytesseract
from PIL import Image, ImageChops

# Gensim
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel

# Flask
import flask
from flask import Flask, render_template, request, make_response, send_from_directory, flash, redirect, url_for, send_file # Converst bytes into a file for downloads
from flask import request, Response
from flask_cors import CORS, cross_origin

# FLask SQLAlchemy, Database
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import desc, asc

basedir = 'sqlite:///' + os.path.join(os.path.abspath(os.path.dirname(__file__)), 'data.sqlite')


app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = basedir
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['SECRET_KEY'] = 'dev'
app.config["DEBUG"] = True
app.config['CORS_HEADERS'] = 'Content-Type'

cors = CORS(app, resources={r"/retrieve": {"origins": "*"}})
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
    webquery = db.Column(db.Text)
    oslog = db.Column(db.String(64))
    pic_date = db.Column(db.DateTime, nullable=False)
    def __repr__(self):
        return f'Pic Name: {self.name} Data: {self.data} text: {self.text} created on: {self.pic_date} location: {self.oslog}'


# Rec table.
class RecContent(db.Model):

    id = db.Column(db.Integer,  primary_key=True)
    userid = db.Column(db.String(64))
    text = db.Column(db.Text)
    rec_date = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)
    def __repr__(self):
        return f'User: {self.userid} created on: {self.rec_date} text: {self.text}'

# Rec table.
class LogContent(db.Model):

    id = db.Column(db.Integer,  primary_key=True)
    userid = db.Column(db.String(64))
    rec_id = db.Column(db.String(64))
    rec_title = db.Column(db.Text)
    rec_url = db.Column(db.Text)
    log_date = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)
    def __repr__(self):
        return f'User: {self.rec_id} created on: {self.log_date} text: {self.rec_title}'

# Index
@app.route('/index', methods=['GET', 'POST'])
@app.route('/')
def index():

    pics = FileContent.query.order_by(desc(FileContent.pic_date)).all()
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

    all_pics = FileContent.query.order_by(desc(FileContent.pic_date)).all()
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
    oslog = request.form['location']
    userid = ''

    newFile = FileContent(name=file.filename, data=data, rendered_data=render_file, text=text, oslog=oslog, userid=userid, pic_date=datetime.utcnow, entities='{}')
    db.session.add(newFile)
    db.session.commit() 
    full_name = newFile.name
    full_name = full_name.split('.')
    file_name = full_name[0]
    file_type = full_name[1]
    file_date = newFile.pic_date
    file_location = newFile.oslog
    file_render = newFile.rendered_data
    file_id = newFile.id
    file_text = newFile.text

    return render_template('upload.html', file_name=file_name, file_type=file_type, file_date=file_date, file_location=file_location, file_render=file_render, file_id=file_id, file_text=file_text)

# upload screens
@app.route('/upload.php', methods=['POST'])
def upload_php():

    file = request.files['image']
    data = file.read()
    render_file = render_picture(data)
    lang = request.form['lang']
    oslog = request.form['extra']
    userid = request.form['username']
    pic_date = filenameToTime(json.loads(request.form['extra'])['filename'])
    # most recent frame
    docs = FileContent.query.filter((FileContent.userid == userid)).order_by(asc(FileContent.pic_date))
    curr = Image.open(BytesIO(data))
    prev = None
    change = None
    isChange = True
    # if sql result is not empty
    if docs.count()>0:
        prev = Image.open(BytesIO(docs[-1].data))
        # if user use the same app --> otherwise user switch app and we run ocr on the entire screen
        if (curr.size == prev.size):
            # only change is extracted
            diff = ImageChops.difference(curr, prev)
            # if there is change, we crop the curr screen to change --> otherwise, no change at all really.
            if (diff.getbbox()):
                print(userid, 'information change', diff.getbbox())
                change = curr.crop((diff.getbbox()))
            else:
                print(userid, 'no change')
                isChange = False
        else:
            print(userid, 'switch app or window size different')

    # convert screen to text - tesseract
    text = ''
    # if there is change, we do ocr
    if isChange:
        text = pytesseract.image_to_string(change, lang=lang) if change else pytesseract.image_to_string(Image.open(file.stream), lang=lang)
        text = text.strip()

    # ibm nlp only there is information change
    entities = detect_entities(text) if text!='' else '{}'
    webquery = getWebQuery(oslog)

    newFile = FileContent(name=file.filename.split('/')[-1], data=data, rendered_data=render_file, text=text, oslog=oslog, userid=userid, pic_date=pic_date, entities=entities, webquery=webquery)
    db.session.add(newFile)
    db.session.commit() 

    return "file uploaded"

# log clicks
@app.route('/logclick', methods=['GET', 'POST'])
def logclick():

    rec_id = request.form['rec_id']
    rec_title = request.form['rec_title']
    rec_url = request.form['rec_url']
    userid = request.form['userid']

    newLog = LogContent(rec_id=rec_id, rec_title=rec_title, rec_url=rec_url, userid=userid)
    db.session.add(newLog)
    db.session.commit() 

    return "logged"

# get logclick
@app.route('/getlogclick')
def getlogclick():

    all_logs = LogContent.query.order_by(asc(LogContent.log_date)).all()
    return all_logs[-1].rec_title

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
    file_data = FileContent.query.filter((FileContent.userid == user_id)).order_by(asc(FileContent.pic_date))
    screens = [screen.text for screen in file_data if screen.text.strip()!='']
    entities = [getEntities(screen.entities) for screen in file_data if screen.text.strip()!='']
    apps = [getApp(screen.oslog) for screen in file_data if screen.text.strip()!='']
    docs = [getDoc(screen.oslog) for screen in file_data if screen.text.strip()!='']
    webqueries = [getWebQuery(screen.oslog) for screen in file_data if screen.text.strip()!='']
    buildCorpus(model_path,screens,entities,apps,docs,webqueries)

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
@app.route('/pic/<int:pic_id>.jpeg')
def pic(pic_id):
    get_pic = FileContent.query.filter_by(id=pic_id).first()
    response = make_response(get_pic.data)
    response.headers.set('Content-Type', 'image/jpeg')
    #response.headers.set(
    #    'Content-Disposition', 'attachment', filename='%s.jpg' % pic_id)
    return response
    
    # # return "data:;base64,"+get_pic.rendered_data
    # return render_template('pic.html', pic=get_pic)

# Update
@app.route('/update/<int:pic_id>', methods=['GET', 'POST'])
def update(pic_id):

    pic = FileContent.query.get(pic_id)

    if request.method == 'POST':
        pic.oslog = request.form['location']
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

# lab
@app.route('/laboratory.php', methods=['POST'])
def predict():

    file = request.files['image']
    data = file.read()
    oslog = request.form['extra']
    userid = request.form['username']
    lang = request.form['lang']
    pic_date = filenameToTime(json.loads(request.form['extra'])['filename'])
    og_img = Image.open(file.stream)

    # crop in case of ui in the chrome screen
    if "chrome" in json.loads(oslog)["appname"].lower():
        f = Image.open(file)
        w,h = f.size
        og_img = f.crop((0, 0, w-300, h))
        buffered = BytesIO()
        og_img.save(buffered, format="JPEG")
        data = buffered.getvalue()

    render_file = render_picture(data)

    # most recent frame
    query_docs = FileContent.query.filter((FileContent.userid == userid)).order_by(asc(FileContent.pic_date))
    docs = [doc for doc in query_docs if doc.text.strip()!='']

    curr = Image.open(BytesIO(data))
    prev = None
    change = None
    isChange = True
    # if sql result is not empty
    # if docs.count()>0:
    if len(docs)>0:
        prev = Image.open(BytesIO(docs[-1].data))
        # if user use the same app --> otherwise user switch app and we run ocr on the entire screen
        if (curr.size == prev.size):
            # only change is extracted
            diff = ImageChops.difference(curr, prev)
            # if there is change, we crop the curr screen to change --> otherwise, no change at all really.
            if (diff.getbbox()):
                print(userid, 'information change', diff.getbbox())
                change = curr.crop((diff.getbbox()))
            else:
                print(userid, 'no change')
                isChange = False
        else:
            print(userid, 'switch app or window size different')

    # convert screen to text - tesseract
    text = ''
    # if there is change, we do ocr
    if isChange:
        text = pytesseract.image_to_string(change, lang=lang) if change else pytesseract.image_to_string(Image.open(file.stream), lang=lang)
        text = text.strip()
    else:
        print(userid, 'dont upload')
        return "file uploaded" 

    # ibm nlp only there is information change
    entities = detect_entities(text) if text!='' else ''
    webquery = getWebQuery(oslog)


    newFile = FileContent(name=file.filename.split('/')[-1], data=data, rendered_data=render_file, text=text, oslog=oslog, userid=userid, pic_date=pic_date, entities=entities, webquery=webquery)
    db.session.add(newFile)
    db.session.commit() 

    # Predict here
    model_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'models/'+userid)
    data = DataLoader(model_path)
    # data.print_info()
    params["corpus_directory"] = model_path
    projector = DataProjector(data, params, model_path)
    projector.generate_latent_space()
    projector.create_feature_matrices()

    pinned_item = []           # the items that are pinned in the frontend (needed for calculating pair similarity)

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

        online_data = docs
        # print(online_data[-1])
        online_screens = [screen.text for screen in online_data[-2:]]
        online_entities = [getEntities(screen.entities) for screen in online_data[-2:]]
        online_apps = [getApp(screen.oslog) for screen in online_data[-2:]]
        online_docs = [getDoc(screen.oslog) for screen in online_data[-2:]]
        online_webqueries = [getWebQuery(screen.oslog) for screen in online_data[-2:]]
        fv_online_docs = getOnlineDocs(model_path, online_screens, online_entities, online_apps, online_docs, online_webqueries)
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


    #organize the recommentations in the right format
    data_output = {}
    data_output["keywords"] = [(sorted_views_list[0][i],data.feature_names[sorted_views_list[0][i]],
                                scored_terms[sorted_views_list[0][i]]) for i in range(min(params["suggestion_count"],data.num_items_per_view[1])) ]
    print("key")
    data_output["applications"] = [(sorted_views_list[1][i],data.feature_names[sorted_views_list[1][i]],
                                    scored_terms[sorted_views_list[1][i]]) for i in range(min(params["suggestion_count"],data.num_items_per_view[2]))]
    print("app")
    data_output["people"] = [(sorted_views_list[2][i],data.feature_names[sorted_views_list[2][i]],
                                scored_terms[sorted_views_list[2][i]] ) for i in range(min(params["suggestion_count"],data.num_items_per_view[3]))]
    print("people")

    data_output["webqueries"] = [(sorted_views_list[3][i],data.feature_names[sorted_views_list[3][i]],
                                scored_terms[sorted_views_list[3][i]] ) for i in range(min(params["suggestion_count"],data.num_items_per_view[4])) if data.feature_names[sorted_views_list[3][i]] != ""]
    print("webqueries")
    # TODO: how many document? I can also send the estimated relevance.
    #data_output["document_ID"] = [(sorted_docs_valid[i],loadLOG(sorted_docs_valid[i])['title'],loadLOG(sorted_docs_valid[i])['url']) for i in range(params["suggestion_count"])]
    #TODO: THis is the hack to distinguish doc and term IDS. Add 600000 to doc IDs for frontend
    # data_output["document_ID"] = [(sorted_docs_valid[i],loadLOG(sorted_docs_valid[i])['title'],loadLOG(sorted_docs_valid[i])['url'],os.path.join(snapshot_directory, "1513349785.60169.jpeg"), loadLOG(sorted_docs_valid[i])['appname']) for i in range(100)]
    data_output["document_ID"] = [(sorted_docs_valid[i],json.loads(docs[sorted_docs_valid[i]].oslog)['title'],json.loads(docs[sorted_docs_valid[i]].oslog)['url'],'../pic/'+str(docs[sorted_docs_valid[i]].id)+'.jpeg',json.loads(docs[sorted_docs_valid[i]].oslog)['appname'],docs[sorted_docs_valid[i]].text) for i in range(50)]
    
    # either this
    data_output["pair_similarity"] = []
    # or this to allow feedback on recs
    # new_recommendations = []
    # for view in range(1, data.num_views):
    #     for i in range(min(params["suggestion_count"],data.num_items_per_view[view])):
    #         new_recommendations.append(sorted_views_list[view-1][i])
    #         if sorted_views_list[view-1][i] not in set(recommended_terms):
    #             recommended_terms.append(sorted_views_list[view-1][i])
    #
    # item_list = list(set(new_recommendations + pinned_item))
    #     # an array to hold the feature vectors
    # recommended_fv = np.empty([len(item_list), projector.num_features])
    # for i in range(len(item_list)):
    #     recommended_fv[i, :] = projector.item_fv(item_list[i]) #get the feature vector
    # #Compute the dot products
    # sim_matrix = np.dot(recommended_fv, recommended_fv.T)
    # #normalize the dot products
    # sim_diags = np.diagonal(sim_matrix)
    # sim_diags = np.sqrt(sim_diags)
    # for i in range(len(item_list)):
    #     sim_matrix[i,:] = sim_matrix[i,:]/sim_diags
    # for i in range(len(item_list)):
    #     sim_matrix[:,i] = sim_matrix[:,i]/sim_diags
    # #save pairwise similarities in a list of tuples
    # all_sims = [(item_list[i],item_list[j],sim_matrix[i,j])
    #             for i in range(len(item_list)-1) for j in range(i+1,len(item_list))]
    # data_output["pair_similarity"] = all_sims

    print("prepared recs")
    newRec = RecContent(userid=userid, text=json.dumps(data_output))
    db.session.add(newRec)
    db.session.commit() 
    return "file uploaded"

# Retrieve docs, webqueries
@app.route('/retrieve/<path:path>')
@cross_origin(origin='*',headers=['Content- Type','Authorization'])
def retrieve(path):
    # userid = 'C1MR3058G940'
    # userid = request.form['username']
    all_recs = RecContent.query.filter_by(userid=path)
    return json.dumps(json.loads(all_recs[-1].text))

# UI
# @app.route('/ui', methods=['GET', 'POST'])
# def ui():
#     return render_template('ui/index.html')
@app.route('/ui/<path:path>')
def send_js(path):
    return send_from_directory('ui', path)
# @app.route('/upload', methods=['POST'])
# def upload():
# 	return {'type':'upload'}

if __name__ == "__main__":
	pass
