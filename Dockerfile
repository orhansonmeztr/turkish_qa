#
FROM python:3.10-slim

#
WORKDIR /code

#
COPY ./requirements.txt /code/requirements.txt

#
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

RUN git clone https://huggingface.co/emrecan/bert-base-turkish-cased-mean-nli-stsb-tr /home/models
RUN git clone https://huggingface.co/sentence-transformers/clip-ViT-B-32-multilingual-v1 /home/models
#
COPY . /code

#
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "3000", "--reload"]