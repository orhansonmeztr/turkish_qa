from dotenv import load_dotenv
import os
import boto3
import shutil
import uuid
from pathlib import Path
from langchain.embeddings import HuggingFaceEmbeddings, CohereEmbeddings, TensorflowHubEmbeddings, OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.retrievers import ContextualCompressionRetriever
from langchain.chains import ConversationalRetrievalChain
from langchain.document_loaders import PyMuPDFLoader
from langchain.retrievers.document_compressors import CohereRerank
from langchain.text_splitter import RecursiveCharacterTextSplitter
from abc import abstractmethod
# from pydantic import BaseModel
# from typing import List, TypedDict
from fastapi import UploadFile

load_dotenv()

aws_access_key_os = os.environ.get("AWS_ACCESS_KEY_OS")
aws_secret_access_key_os = os.environ.get("AWS_SECRET_ACCESS_KEY_OS")
openai_api_key = os.environ.get("OPENAI_API_KEY")
cohere_api_key = os.environ.get("COHERE_API_KEY")
chunk_size = 500
chunk_overlap = 30
bucket_name = "turkish-qa-collections"
local_temp_folder = "temp"


# class ReqBody(BaseModel):
#     embed_model_number: int = 0
#     top_k: int = 10
#     top_n: int = 3
#     engine_name: str = 'gpt-3.5-turbo'
#     llm_temp: float = '0.0'
#     reduction_type: str = 'map_reduce'
#     text_input: str = "Gece çalışması nedir"


# class Feature(TypedDict):
#     result_status: bool
#     answer: str
#     sources: list


# class AnsOut(BaseModel):
#     features: List[Feature]


def upload_process_send_s3(uploaded_file: UploadFile, collection_id: str, embedding_model_number: int):
    file_name = str(uuid.uuid1()) + Path(uploaded_file.filename).suffix
    # old_file_name = uploaded_file.filename
    uploaded_file.filename = local_temp_folder + '/' + file_name
    res = True
    result = {"file_name": file_name}
    object_path = os.path.join(os.getcwd(), local_temp_folder, file_name)
    if os.path.exists(object_path):
        print(f"The file with the name '{file_name}' already exists in the local folder.")
        res = False
        result["message"] = f"The file with the name '{file_name}' already exists in the local folder."
    else:
        try:
            with open(uploaded_file.filename, 'wb') as f:
                shutil.copyfileobj(uploaded_file.file, f)
        except Exception as e:
            res = False
            result["error_message"] = f"An error occurred: '{e}'"
        finally:
            uploaded_file.file.close()
    if res:
        res1 = process_pdf_send_s3(local_file=file_name,
                                   collection_id=collection_id,
                                   embedding_model_number=embedding_model_number)["result_status"]
        res = res and res1
        res2 = upload_file_to_s3(local_file=file_name,
                                 collection_id=collection_id,
                                 s3_file_name=file_name)["result_status"]
        res = res and res2
    result["result_status"] = res
    return result


# Base class for Document Processing
class DocumentProcessor:
    @abstractmethod
    def process_document(self):
        raise NotImplementedError

    @abstractmethod
    def split_to_chunks(self):
        raise NotImplementedError


class PDFDocumentProcessor(DocumentProcessor):
    def __init__(self, local_file):
        self.local_file_path = os.path.join(os.getcwd(), local_temp_folder, local_file)
        self.data = None
        self.docs = None

    def process_document(self):
        loader = PyMuPDFLoader(self.local_file_path)
        dat = loader.load()
        for i in range(len(dat)):
            dat[i].page_content = dat[i].page_content.replace("\n", " ")
        self.data = dat

    def split_to_chunks(self):
        splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size,
                                                  chunk_overlap=chunk_overlap,
                                                  length_function=len)
        docs = splitter.split_documents(self.data)
        self.docs = docs


def select_model(embedding_model_number=0):
    if embedding_model_number == 0:
        embedding_model = HuggingFaceModel(model_name_or_path="bert-base-turkish-cased-mean-nli-stsb-tr")
    elif embedding_model_number == 1:
        embedding_model = TensorflowModel()
    elif embedding_model_number == 2:
        embedding_model = CohereModel()
    elif embedding_model_number == 3:
        embedding_model = HuggingFaceModel(model_name_or_path="clip-ViT-B-32-multilingual-v1")
    elif embedding_model_number == 4:
        embedding_model = OpenaiModel()
    else:
        return False  # Data other than 0-4 came.

    return embedding_model


class HuggingFaceModel:  # for emrecan and clip models
    def __init__(self, model_name_or_path):   # , device="cpu"
        self.model = HuggingFaceEmbeddings(model_name=model_name_or_path)   # , model_kwargs={'device': device}


class CohereModel:
    def __init__(self, model_name_or_path='embed-multilingual-v2.0'):
        self.model = CohereEmbeddings(model=model_name_or_path, cohere_api_key=cohere_api_key)


class TensorflowModel:
    def __init__(self, model_url="https://tfhub.dev/google/universal-sentence-encoder-multilingual/3"):
        self.model = TensorflowHubEmbeddings(model_url=model_url)


class OpenaiModel:
    def __init__(self, model_name="gpt-3.5-turbo"):
        self.model = OpenAIEmbeddings(model_name=model_name, openai_api_key=openai_api_key)


class FAISSIndexManager:
    def __init__(self, embedding_model):
        self.embedding_model = embedding_model
        self.dbFAISS = None

    def create_index_db(self, docs):
        dbFAISS = FAISS.from_documents(docs, self.embedding_model)
        self.dbFAISS = dbFAISS

    def save_index_to_s3(self, collection_id, folder_name):
        result = {}
        res: bool
        if not self.dbFAISS:
            result["faiss_error"] = "No FAISS index to save."
            res = False
        else:
            object_path = os.path.join(os.getcwd(), local_temp_folder, folder_name)
            self.dbFAISS.save_local(object_path)
            s3 = S3client()
            try:
                res = s3.upload_folder(object_path, collection_id, folder_name)["result_status"]
            except Exception as e:
                res = False
                result["error_message"] = f"An error occurred: '{e}'"
            del s3
            shutil.rmtree(object_path)
        result["result_status"] = res
        return result


class FAISSIndexRetriever:
    def __init__(self, embedding_model, top_K: int = 10):
        self.embedding_model = embedding_model
        self.dbFAISS = None
        self.top_K = top_K
        self.retriever = None

    def load_index_from_local(self, collection_id, local_folder_name):
        result = {}
        res = True
        object_path = os.path.join(os.getcwd(), local_temp_folder, collection_id, local_folder_name)
        if not os.path.exists(object_path):
            result["faiss_error"] = (f"The index folder with the name '{collection_id}/{local_folder_name}' was not "
                                     f"found in the local folder.")
            res = False
        else:
            self.dbFAISS = FAISS.load_local(object_path, self.embedding_model)
            self.retriever = self.dbFAISS.as_retriever(search_kwargs={"k": self.top_K})
        result["result_status"] = res
        return result

    def load_indexes_from_local_collection(self, collection_id):
        result = {}
        res = True
        root_dir = os.path.join(os.getcwd(), local_temp_folder, collection_id)
        if not os.path.exists(root_dir):
            result["faiss_error"] = (f"The collection folder with the name '{collection_id}' that contains indexes was "
                                     f"not found in the local folder.")
            res = False
        else:
            index_folders = []

            for file in os.listdir(root_dir):
                d = os.path.join(root_dir, file)
                if os.path.isdir(d):
                    index_folders.append(d)
            if len(index_folders) == 0:
                res = False
                result["index_error"] = f"There is no any index folder in the given local '{collection_id}' folder."
            else:
                db = FAISS.load_local(index_folders[0], self.embedding_model)
                for i in range(1, len(index_folders)):
                    db1 = FAISS.load_local(index_folders[i], self.embedding_model)
                    db.merge_from(db1)
                dbFAISS = db
                self.dbFAISS = dbFAISS
                self.retriever = self.dbFAISS.as_retriever(search_kwargs={"k": self.top_K})
        result["result_status"] = res
        return result


class S3client:
    def __init__(self):
        self.bucket_name = bucket_name
        self.client = boto3.client(service_name="s3", aws_access_key_id=aws_access_key_os,
                                   aws_secret_access_key=aws_secret_access_key_os)

    def check_object_exists_on_s3(self, file_or_folder_name):
        response = self.client.list_objects_v2(Bucket=self.bucket_name, Prefix=file_or_folder_name)
        for item in response.get('Contents', []):
            if item['Key'] == file_or_folder_name + '/':
                return True
        return False

    # def folder_names_in_a_collection_on_s3(self, collection_id: str):
    #     response = self.client.list_objects_v2(Bucket=self.bucket_name)
    #     folders = []
    #     for obj in response['Contents']:
    #         if obj['Key'].endswith('/') and obj['Key'] != collection_id + '/':
    #             sub_folder_in_collection = obj['Key'].split("/")[1]
    #             if sub_folder_in_collection not in folders:
    #                 folders.append(sub_folder_in_collection)
    #     return folders

    def upload_file_(self, local_file_path, collection_id, s3_file_name):
        res = True
        result = {}
        s3_file_path = collection_id + "/" + s3_file_name
        if self.check_object_exists_on_s3(s3_file_path):
            result["file_error"] = f"The file with the name '{s3_file_path}' already exists in the collection."
            res = False
        else:
            try:
                self.client.upload_file(local_file_path, self.bucket_name, collection_id + "/" + s3_file_name)
            except Exception as e:
                result["upload_err"] = f"An error occurred: '{e}'"
                res = False
        result["result_status"] = res
        return result

    def upload_folder(self, local_folder_path, collection_id, s3_folder_name):
        res = True
        result = {}
        try:
            if not self.check_object_exists_on_s3(collection_id):
                self.client.put_object(Bucket=self.bucket_name, Body='', Key=collection_id + "/")
            if not self.check_object_exists_on_s3(collection_id + "/" + s3_folder_name):
                self.client.put_object(Bucket=self.bucket_name, Body='', Key=collection_id + "/" + s3_folder_name + "/")
            for file_name in os.listdir(local_folder_path):
                file_path = os.path.join(local_folder_path, file_name)
                self.client.upload_file(file_path,
                                        self.bucket_name,
                                        collection_id + "/" + s3_folder_name + "/" + file_name)
        except Exception as e:
            result["upload_err"] = f"An error occurred: '{e}'"
            res = False
        result["result_status"] = res
        return result

    def get_file_folders(self, collection_id):
        file_names = []
        folders = []
        paginator = self.client.get_paginator('list_objects_v2')
        response_iterator = paginator.paginate(Bucket=self.bucket_name, Prefix=collection_id + "/")
        for response in response_iterator:
            for item in response.get('Contents', []):
                key = item['Key']
                if key[-1] == "/":
                    folders.append(key)
                else:
                    if str(key).endswith(".faiss") or str(key).endswith(".pkl"):
                        file_names.append(key)
        return file_names, folders

    def download_index_to_local(self, collection_id):
        res = True
        result = {}
        file_names, folders = self.get_file_folders(collection_id)
        local_path = Path(os.path.join(os.getcwd(), local_temp_folder))
        try:
            for folder in folders:
                folder_path = Path.joinpath(local_path, folder)
                if os.path.exists(folder_path):
                    shutil.rmtree(folder_path)
                folder_path.mkdir(parents=True, exist_ok=True)

            for file_name in file_names:
                if not str(file_name).endswith(".pdf"):
                    file_path = Path.joinpath(local_path, file_name)
                    self.client.download_file(
                        self.bucket_name,
                        file_name,
                        str(file_path)
                    )
        except Exception as e:
            result["download_err"] = f"An error occurred: '{e}'"
            res = False
        result["result_status"] = res
        return result


class CohereReranker:
    def __init__(self, model_name_or_path='rerank-multilingual-v2.0', top_n=3):
        self.model_name_or_path = model_name_or_path
        self.top_n = top_n
        self.reranker = CohereRerank(model=self.model_name_or_path, top_n=self.top_n)


class OpenAILLMInteraction:
    def __init__(self,
                 base_retriever: FAISSIndexRetriever,
                 compressor: CohereReranker,
                 model_name="gpt-3.5-turbo",
                 temperature=0.0,
                 reduction_type="map_reduce"):
        self.llm = ChatOpenAI(openai_api_key=openai_api_key, temperature=temperature, model_name=model_name,
                              request_timeout=120)
        self.reduction_type = reduction_type
        self.retriever = ContextualCompressionRetriever(
            base_compressor=compressor.reranker,
            base_retriever=base_retriever)
        self.ret_chain = ConversationalRetrievalChain.from_llm(self.llm,
                                                               retriever=self.retriever,
                                                               chain_type=self.reduction_type,
                                                               return_source_documents=True)

    def return_results(self, question: str, chat_history=None):
        result = {}
        res = True
        response = None
        if chat_history is None:
            chat_history = []
        chat_history = chat_history
        try:
            response = self.ret_chain({"question": question, "chat_history": chat_history})
        except Exception as e:
            result["error_message"] = f"Error getting results from OpenAI: {e}"
            res = False
        if res:
            chat_history.append((question, response['answer']))
            sources = [{'ref_text': response['source_documents'][i].page_content.replace('\n', ''),
                        'ref_file_name': Path(response['source_documents'][i].metadata['source']).name,
                        'ref_page': str(int(response['source_documents'][i].metadata['page']) + 1)}
                       for i in range(len(response['source_documents']))
                       ]
            result["answer"] = response['answer']
            result["sources"] = sources
        result["result_status"] = res
        return result


def upload_file_to_s3(local_file, collection_id, s3_file_name):
    local_file_path = os.path.join(os.getcwd(), local_temp_folder, local_file)
    s3 = S3client()
    result = {}
    res = True
    try:
        s3.upload_file_(local_file_path=local_file_path,
                        collection_id=collection_id,
                        s3_file_name=s3_file_name)
    except Exception as e:
        res = False
        result["error_message"] = f"An error occurred: '{e}'"
    del s3
    if res:
        del_local_file(local_file)
    result["result_status"] = res
    return result


def del_local_file(local_file_name):
    file_ = os.path.join(os.getcwd(), local_temp_folder, local_file_name)
    res = True
    err_msg = None
    if not os.path.exists(file_):
        print(f"The file with the name '{local_file_name}' not found in the local folder.")
        res = False
    else:
        try:
            os.remove(file_)
        except Exception as e:
            err_msg = e
    result = {"result_status": res}
    if not res:
        result["message"] = f"The file with the name '{local_file_name}' not found in the local folder."
    if err_msg:
        result["error_message"] = f"An error occurred: '{err_msg}'"
    return result


def del_local_folder(local_folder_name):
    folder_ = os.path.join(os.getcwd(), local_temp_folder, local_folder_name)
    res = True
    err_msg = None
    if not os.path.exists(folder_):
        print(f"The folder with the name '{local_folder_name}' not found in the local folder.")
        res = False
    else:
        try:
            shutil.rmtree(folder_)
        except Exception as e:
            err_msg = e
    result = {"result_status": res}
    if not res:
        result["message"] = f"The folder with the name '{local_folder_name}' not found in the local folder."
    if err_msg:
        result["error_message"] = f"An error occurred: '{err_msg}'"
    return result


def process_pdf_send_s3(local_file: str,
                        collection_id: str,
                        embedding_model_number: int):
    pdfDocumentProcessor = PDFDocumentProcessor(local_file=local_file)
    pdfDocumentProcessor.process_document()
    pdfDocumentProcessor.split_to_chunks()
    doc_chunks = pdfDocumentProcessor.docs
    embedding_model = select_model(embedding_model_number=embedding_model_number).model
    faissIndexManager = FAISSIndexManager(embedding_model=embedding_model)
    faissIndexManager.create_index_db(docs=doc_chunks)
    res = faissIndexManager.save_index_to_s3(collection_id=collection_id,
                                             folder_name=Path(local_file).stem)["result_status"]
    result = {"result_status": res}
    return result


def download_collection_from_s3_to_local(collection_id):
    result = {}
    s3 = S3client()
    res = s3.download_index_to_local(collection_id=collection_id)["result_status"]
    result["result_status"] = res
    return result


def ask_to_llm_with_local_collection(collection_id: str,
                                     embedding_model_number: int,
                                     top_K: int,
                                     top_n: int,
                                     llm: str,
                                     engine_name: str,
                                     temperature: float,
                                     reduction_type: str,
                                     question: str):
    result = {}
    if llm == "openai":
        embeddings = select_model(embedding_model_number=embedding_model_number).model
        faissIndexRetriever = FAISSIndexRetriever(embedding_model=embeddings, top_K=top_K)
        res = faissIndexRetriever.load_indexes_from_local_collection(collection_id=collection_id)
        result["result_status"] = res["result_status"]
        if result["result_status"]:
            retriever = faissIndexRetriever.retriever
            compressor = CohereReranker(model_name_or_path='rerank-multilingual-v2.0', top_n=top_n)
            openai_llm = OpenAILLMInteraction(base_retriever=retriever,
                                              compressor=compressor,
                                              model_name=engine_name,
                                              temperature=temperature,
                                              reduction_type=reduction_type)
            response = openai_llm.return_results(question=question)
            result["result_status"] = response["result_status"]
            del response["result_status"]
            result["response"] = response
    return result
