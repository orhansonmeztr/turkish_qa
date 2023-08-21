#
FROM python:3.10-slim

#
WORKDIR /code

#
COPY ./requirements.txt /code/requirements.txt
RUN apt-get update

# Install software
RUN apt-get install git
RUN apt-get install git-lfs
RUN apt-get install curl
#
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

RUN git lfs clone https://huggingface.co/emrecan/bert-base-turkish-cased-mean-nli-stsb-tr
RUN git lfs clone https://huggingface.co/sentence-transformers/clip-ViT-B-32-multilingual-v1

RUN mkdir universal-sentence-encoder-multilingual_3
RUN curl -o universal-sentence-encoder-multilingual_3.tar.gz https://storage.googleapis.com/tfhub-modules/google/universal-sentence-encoder-multilingual/3.tar.gz
RUN tar -xvzf universal-sentence-encoder-multilingual_3.tar.gz -C universal-sentence-encoder-multilingual_3/
RUN rm -rf universal-sentence-encoder-multilingual_3.tar.gz
#
COPY . /code

#
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "3000", "--reload"]