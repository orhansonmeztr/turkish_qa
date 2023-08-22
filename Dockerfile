#
FROM python:3.10-slim

#
WORKDIR /code

#
COPY ./requirements.txt /code/requirements.txt

RUN apt-get update && apt-get install -y wkhtmltopdf
#
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt
#

#
COPY . /code

#
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "3000", "--reload"]
