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
from datetime import datetime, timedelta
from pathlib import Path
import shutil

# Tesseract
# import pytesseract
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
from sqlalchemy import desc, asc, or_

basedir = 'sqlite:///' + os.path.join(os.path.abspath(os.path.dirname(__file__)), 'data1.sqlite')


app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = basedir
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['SECRET_KEY'] = 'dev'
app.config["DEBUG"] = True
app.config['CORS_HEADERS'] = 'Content-Type'

cors = CORS(app, resources={r"/retrieve": {"origins": "*"}, r"/logclick": {"origins": "*"}})
db = SQLAlchemy(app)
allDataLoad = {}
# model_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'models/C1MR3058G940')



# Picture table. By default the table name is filecontent
class FileContent(db.Model):

    """ 
    The first time the app runs you need to create the table. In Python
    terminal import db, Then run db.create_all()
    """
    """ ___tablename__ = 'yourchoice' """ # You can override the default table name

    id = db.Column(db.Integer,  primary_key=True)
    name = db.Column(db.String(128), nullable=False)
    data = db.Column(db.LargeBinary, nullable=False, default=None) 
    rendered_data = db.Column(db.Text, nullable=False, default=None)
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


def update():
    pic_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'pics/FA3441DEC434')
    Path(pic_path).mkdir(parents=True, exist_ok=True)
    pics = FileContent.query.filter_by(userid=='F83441DEC435')
    for pic in pics:
        print(pic.name)
        curr = Image.open(BytesIO(pic.data))
        curr_img_fname = pic_path+'/'+json.loads(pic.oslog)['filename']+'.jpeg'
        curr.save(curr_img_fname)
        # pic.name = 'aaaaaa'
    # db.session.commit()
    # if request.method == 'POST':
    #     pic.oslog = request.form['location']
    #     pic.text = request.form['text']

    #     db.session.commit()
    #     flash(f'{pic.name} Has been updated')
    #     return redirect(url_for('index'))
    # return render_template('update.html', pic=pic)

update()
