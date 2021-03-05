from setuptools import setup, find_packages

requires = [
    'python-dotenv',
    'flask',
    'jsonify',
    'pandas',
    'spacy',
    'nltk',
    'en_core_web_sm @ https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.3.1/en_core_web_sm-2.3.1.tar.gz',
    'gensim',
    'numpy',
    'sklearn',
    'flask-sqlalchemy',
    'flask-marshmallow',
    'marshmallow-sqlalchemy',
    'flask-migrate',
    'pytesseract',
    'Pillow',
    'ibm-watson'
]

setup(
    name='flask_main',
    version='0.1',
    description='coadapt project apache2 license',
    author='Vuong',
    author_email='vuong@cs.helsinki.fi',
    keywords='conversational agent, user modeling',
    packages=find_packages(),
    include_package_data=True,
    install_requires=requires
)
