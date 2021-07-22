FROM python:3.9-slim-buster

WORKDIR /app

COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt

ADD model model
COPY preprocess.py preprocess.py
COPY model.py model.py
COPY app.py app.py

EXPOSE 80

CMD [ "python3", "app.py", "--host=0.0.0.0", "--port=80"]