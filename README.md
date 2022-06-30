# contextpredict
Coadapt project


sudo python3 -m pip install -e .

#edit host in .env
nano .env

cd main

cp data.sqlite_template data.sqlite

sudo python3 -m flask run

sudo python3 -m flask run --cert=cert.pem --key=key.pem

Install tesseract 4.1.1 or 5.0

https://tesseract-ocr.github.io/tessdoc/Installation.html
