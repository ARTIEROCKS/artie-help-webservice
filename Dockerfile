FROM python:3.9-slim-buster

WORKDIR /app

COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt

ADD model model
ADD service service
ADD repository repository
COPY app.py app.py

EXPOSE 5000

CMD [ "python3", "app.py", "--host=0.0.0.0", "--port=5000"]