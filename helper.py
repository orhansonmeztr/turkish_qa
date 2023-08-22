from dotenv import load_dotenv
import os
import boto3
import shutil
import uuid
import pdfkit
from pathlib import Path
from langchain.embeddings import HuggingFaceEmbeddings, HuggingFaceHubEmbeddings
from langchain.embeddings import CohereEmbeddings, TensorflowHubEmbeddings, OpenAIEmbeddings
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
huggingface_api_token = os.environ.get("HUGGINGFACEHUB_API_TOKEN")
chunk_size = 500
chunk_overlap = 30
bucket_name = "turkish-qa-collections"
local_temp_folder = "temp"
total_number_of_embedding_models = 5
total_number_of_collections_in_local = 5
supported_file_types_for_uploading = ["pdf"]

temp_dir = os.path.join(os.getcwd(), local_temp_folder)
if not os.path.exists(temp_dir):
    os.makedirs(temp_dir)


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

def select_model(embed_model_number=4):
    if embed_model_number == 0:
        embedding_model = HuggingFaceEmbeddings(
            model_name="emrecan/bert-base-turkish-cased-mean-nli-stsb-tr",
            model_kwargs={'device': "cpu"})
    elif embed_model_number == 1:
        embedding_model = TensorflowHubEmbeddings(
            model_url="https://tfhub.dev/google/universal-sentence-encoder-multilingual/3")
    elif embed_model_number == 2:
        embedding_model = CohereEmbeddings(model="embed-multilingual-v2.0", cohere_api_key=cohere_api_key)
    elif embed_model_number == 3:
        embedding_model = HuggingFaceHubEmbeddings(
            huggingfacehub_api_token=huggingface_api_token,
            repo_id="sentence-transformers/clip-ViT-B-32-multilingual-v1")
    elif embed_model_number == 4:
        embedding_model = OpenAIEmbeddings(model="text-embedding-ada-002", openai_api_key=openai_api_key)
    else:
        return False
    return embedding_model


def process_file(flag: bool, local_file: str, filetype: str, collection_id: str) -> bool:
    res = flag
    doc_chunks = produce_doc_chunks_from_file(local_file=local_file, filetype=filetype)
    for i in [2, 3, 4]:  # range(total_number_of_embedding_models):
        res1 = process_file_send_s3(local_file=local_file,
                                    collection_id=collection_id,
                                    embed_model_number=i,
                                    doc_chunks=doc_chunks)["result_status"]
        res = res and res1
    if res:
        res2 = upload_file_to_s3(local_file=local_file,
                                 collection_id=collection_id,
                                 s3_file_name=local_file)["result_status"]
        res = res and res2
    del_local_file(local_file)
    return res


def produce_pdf_from_url_to_s3(url: str, collection_id: str):
    res = True
    result = {"result_status": res}
    file_name = str(uuid.uuid1()) + ".pdf"
    filetype = str(Path(file_name).suffix).split(".")[-1]
    file_path = os.path.join(os.getcwd(), local_temp_folder, file_name)
    if url is None or url == "" or (not url.startswith("http")):
        res = False
        result["error_message"] = f"Only files with the extension {supported_file_types_for_uploading} are allowed."
    else:
        try:
            # config = pdfkit.configuration(wkhtmltopdf=r"C:\Program Files\wkhtmltopdf\bin\wkhtmltopdf.exe")
            config = pdfkit.configuration(wkhtmltopdf="/usr/bin/wkhtmltopdf")
            pdf = pdfkit.from_url(url, False, configuration=config)
            with open(file_path, mode="wb") as f:
                f.write(pdf)
            result["file_name"] = file_name
        except Exception as e:
            res = False
            result["message"] = f"An error occurred: {e}"
        if res:
            res1 = process_file(flag=res, local_file=file_name, filetype=filetype, collection_id=collection_id)
            res = res and res1
        if res:
            shutil.rmtree(os.path.join(os.getcwd(), local_temp_folder, Path(file_name).stem))

    result["result_status"] = res
    return result


def upload_file_process_send_s3(uploaded_file: UploadFile, collection_id: str):
    filetype = str(Path(uploaded_file.filename).suffix).split(".")[-1]
    result = {}
    if filetype in supported_file_types_for_uploading:
        file_name = str(uuid.uuid1()) + "." + filetype
        old_file_name = uploaded_file.filename
        uploaded_file.filename = local_temp_folder + '/' + file_name
        res = True
        result["old_file_name"] = old_file_name
        result["new_file_name"] = file_name
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
            res1 = process_file(flag=res, local_file=file_name, filetype=filetype, collection_id=collection_id)
            res = res and res1
        if res:
            shutil.rmtree(os.path.join(os.getcwd(), local_temp_folder, Path(file_name).stem))
    else:
        result["error_message"] = f"Only files with the extension {supported_file_types_for_uploading} are allowed."
        res = False
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
            s3 = S3()
            try:
                res = s3.upload_folder(object_path,
                                       collection_id,
                                       folder_name)["result_status"]
            except Exception as e:
                res = False
                result["error_message"] = f"An error occurred: '{e}'"
            del s3
            shutil.rmtree(object_path)
        result["result_status"] = res
        return result


class FAISSIndexRetriever:
    def __init__(self, embedding_model, top_k: int = 10):
        self.embedding_model = embedding_model
        self.dbFAISS = None
        self.top_k = top_k
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
            self.retriever = self.dbFAISS.as_retriever(search_kwargs={"k": self.top_k})
        result["result_status"] = res
        return result

    def load_indexes_from_local_collection(self, collection_id, embed_model_number: int):
        result = {}
        res = True
        root_dir = os.path.join(os.getcwd(), local_temp_folder, collection_id)
        if not os.path.exists(root_dir):
            result["faiss_error"] = (f"The collection folder with the name '{collection_id}' was not found in the "
                                     f"local folder.")
            res = False
        else:
            index_folders = []
            for file in os.listdir(root_dir):
                doc = os.path.join(root_dir, file)
                if os.path.isdir(doc):
                    for index_folder in os.listdir(doc):
                        if index_folder == str(embed_model_number):
                            index_folders.append(os.path.join(root_dir, file, index_folder))
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
                self.retriever = self.dbFAISS.as_retriever(search_kwargs={"k": self.top_k})
        result["result_status"] = res
        return result


class S3:
    def __init__(self):
        self.bucket_name = bucket_name
        self.client = boto3.client(service_name="s3", aws_access_key_id=aws_access_key_os,
                                   aws_secret_access_key=aws_secret_access_key_os)

    def upload_file_(self, local_file_path, collection_id, s3_file_name):
        res = True
        result = {}
        s3_file_path = collection_id + "/" + s3_file_name
        file_names, folders = self.get_file_folder_names(prefix=collection_id + "/")

        if s3_file_path in file_names:
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
        file_names, folders = self.get_file_folder_names(prefix=collection_id + "/")
        try:
            if not collection_id + "/" in folders:
                self.client.put_object(Bucket=self.bucket_name, Body='', Key=collection_id + "/")
            if not collection_id + "/" + s3_folder_name in folders:
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

    def get_file_folder_names(self, prefix):
        file_names = []
        folders = []
        paginator = self.client.get_paginator('list_objects_v2')
        response_iterator = paginator.paginate(Bucket=self.bucket_name, Prefix=prefix)
        for response in response_iterator:
            for item in response.get('Contents', []):
                key = item['Key']
                if key[-1] == "/":
                    folders.append(key)
                else:
                    file_names.append(key)
        return file_names, folders

    def del_doc_with_index_from_collection(self, collection_id: str, file_name_with_extension: str):
        res = True
        result = {}
        file = collection_id + "/" + file_name_with_extension
        try:
            self.client.delete_object(Bucket=self.bucket_name, Key=file)
        except Exception as e:
            res = False
            result["error_message"] = f"An error occurred: '{e}'"

        folder = collection_id + "/" + file_name_with_extension.split(".")[0] + "/"
        try:
            files_in_folder = self.client.list_objects_v2(Bucket=self.bucket_name, Prefix=folder)["Contents"]
            files_to_delete = []
            for f in files_in_folder:
                files_to_delete.append({"Key": f["Key"]})
            self.client.delete_objects(Bucket=self.bucket_name, Delete={"Objects": files_to_delete})
        except Exception as e:
            res = False
            result["error_message"] = f"An error occurred: '{e}'"

        result["result_status"] = res
        return result

    def del_collection(self, collection_id: str):
        res = True
        result = {}
        try:
            files_in_folder = self.client.list_objects_v2(Bucket=self.bucket_name,
                                                          Prefix=collection_id + "/")["Contents"]
            files_to_delete = []
            for f in files_in_folder:
                files_to_delete.append({"Key": f["Key"]})
            self.client.delete_objects(Bucket=self.bucket_name, Delete={"Objects": files_to_delete})
        except Exception as e:
            res = False
            result["error_message"] = f"An error occurred: '{e}'"

        result["result_status"] = res
        return result

    def download_index_to_local(self, collection_id):
        res = True
        result = {}
        s3_file_names, s3_folders = self.get_file_folder_names(prefix=collection_id + "/")
        local_path = Path(os.path.join(os.getcwd(), local_temp_folder))
        s3_doc_folders = []
        for fol in s3_folders:
            li = str(fol).split("/")
            if len(li) == 4:
                fol_name = li[1]
                if fol_name not in s3_doc_folders:
                    s3_doc_folders.append(fol_name)

        for obj in os.listdir(Path.joinpath(local_path, collection_id)):
            doc = os.path.join(local_path, collection_id, obj)
            if os.path.isdir(doc):
                if obj not in s3_doc_folders:
                    shutil.rmtree(doc)

        try:
            for folder in s3_folders:
                folder_path = Path.joinpath(local_path, folder)
                folder_path.mkdir(parents=True, exist_ok=True)

            for file_name in s3_file_names:
                if not str(file_name).split(".")[-1] in supported_file_types_for_uploading:
                    file_path = Path.joinpath(local_path, file_name)
                    if not os.path.exists(file_path):
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
                 llm_temp=0.0,
                 reduction_type="map_reduce"):
        self.llm = ChatOpenAI(openai_api_key=openai_api_key, temperature=llm_temp, model_name=model_name,
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
    s3 = S3()
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
    result["result_status"] = res
    return result


def organize_collection_folders_in_local():
    root_dir = os.path.join(os.getcwd(), local_temp_folder)
    sub_folders = []
    creat_dates = []
    result = {}
    res = True
    for obj in os.listdir(root_dir):
        doc = os.path.join(root_dir, obj)
        if os.path.isdir(doc):
            sub_folders.append(doc)
            creat_dates.append(os.path.getctime(doc))
    if len(sub_folders) > total_number_of_collections_in_local:
        K = sorted(range(len(creat_dates)), key=lambda i: creat_dates[i])
        indexes_for_del = K[:-total_number_of_collections_in_local]
        for j in range(len(indexes_for_del)):
            try:
                shutil.rmtree(sub_folders[indexes_for_del[j]])
            except Exception as e:
                result["error_message"] = f"An error occurred: '{e}'"
                res = False
                break
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


def produce_doc_chunks_from_file(local_file: str, filetype: str):
    doc_chunks = None
    if filetype in supported_file_types_for_uploading:
        if filetype == "pdf":
            pdfDocumentProcessor = PDFDocumentProcessor(local_file=local_file)
            pdfDocumentProcessor.process_document()
            pdfDocumentProcessor.split_to_chunks()
            doc_chunks = pdfDocumentProcessor.docs
    return doc_chunks


def process_file_send_s3(local_file: str,
                         collection_id: str,
                         embed_model_number: int,
                         doc_chunks):
    embedding_model = select_model(embed_model_number=embed_model_number)
    faissIndexManager = FAISSIndexManager(embedding_model=embedding_model)
    faissIndexManager.create_index_db(docs=doc_chunks)
    res = faissIndexManager.save_index_to_s3(collection_id=collection_id,
                                             folder_name=Path(local_file).stem + "/" + str(embed_model_number)
                                             )["result_status"]
    result = {"result_status": res}
    return result


def download_collection_from_s3_to_local(collection_id):
    result = {}
    s3 = S3()
    res = s3.download_index_to_local(collection_id=collection_id)["result_status"]
    result["result_status"] = res
    return result


def del_doc_with_index_from_s3_collection(collection_id: str, file_name_with_extension: str):
    result = {}
    res = True
    if collection_id is None or file_name_with_extension is None:
        res = False
        result["message"] = "collection_id and file_name_with_extension must be given"
    else:
        s3 = S3()
        res1 = s3.del_doc_with_index_from_collection(collection_id=collection_id,
                                                     file_name_with_extension=file_name_with_extension)["result_status"]
        res = res and res1
    result["result_status"] = res
    return result


def del_s3_collection(collection_id: str):
    result = {}
    res = True
    if collection_id is None:
        res = False
        result["message"] = "collection_id must be given"
    else:
        s3 = S3()
        res1 = s3.del_collection(collection_id=collection_id)["result_status"]
        res = res and res1
    result["result_status"] = res
    return result


def ask_to_llm_with_local_collection(collection_id: str,
                                     embed_model_number: int,
                                     top_k: int,
                                     top_n: int,
                                     llm: str,
                                     engine_name: str,
                                     llm_temp: float,
                                     reduction_type: str,
                                     question: str):
    result = {}
    organize_collection_folders_in_local()
    res = download_collection_from_s3_to_local(collection_id)["result_status"]
    result["result_status"] = res
    if res:
        if llm == "openai":
            embeddings = select_model(embed_model_number=embed_model_number)
            faissIndexRetriever = FAISSIndexRetriever(embedding_model=embeddings, top_k=top_k)
            res1 = faissIndexRetriever.load_indexes_from_local_collection(collection_id=collection_id,
                                                                          embed_model_number=embed_model_number)
            result["result_status"] = res1["result_status"]
            if result["result_status"]:
                retriever = faissIndexRetriever.retriever
                compressor = CohereReranker(model_name_or_path='rerank-multilingual-v2.0', top_n=top_n)
                openai_llm = OpenAILLMInteraction(base_retriever=retriever,
                                                  compressor=compressor,
                                                  model_name=engine_name,
                                                  llm_temp=llm_temp,
                                                  reduction_type=reduction_type)
                response = openai_llm.return_results(question=question)
                result["result_status"] = response["result_status"]
                del response["result_status"]
                result["response"] = response
    return result
