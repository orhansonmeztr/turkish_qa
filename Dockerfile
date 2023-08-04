#
FROM python:3.19

#
WORKDIR /code

#
COPY ./requirements.txt /code/requirements.txt

#
RUN pip install boto3
RUN pip install langchain
RUN pip install cohere
RUN pip install python-dotenv
RUN pip install fastapi
RUN pip install uvicorn
RUN pip install python-multipart
RUN pip install PyMuPDF
RUN pip install sentence-transformers
RUN pip install faiss-cpu
RUN pip install openai
RUN pip install tiktoken
RUN pip install streamlit

#
COPY . /code

#
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "3000", "--reload"]
