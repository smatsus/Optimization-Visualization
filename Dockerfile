FROM jupyter/datascience-notebook
WORKDIR /app

EXPOSE 8888
COPY . /app
# RUN pip3 install -r requirements.txt 
