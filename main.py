import helper
from fastapi import FastAPI, UploadFile

app = FastAPI()


@app.get('/api/hc')
def health_check():
    return {'status': 'ok'}


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.post("/produce_pdf_from_url_to_s3")
def produce_pdf_from_url_to_s3(url: str = "https://en.wikipedia.org/wiki/Large_language_model",
                               collection_id: str = "378a73bc-e0cb-4777-a58d-cc6913552b45"):
    return helper.produce_pdf_from_url_to_s3(url=url, collection_id=collection_id)


@app.post("/upload_file_process_send_s3")
def upload_file_process_send_s3(uploaded_file: UploadFile,
                                collection_id: str = "378a73bc-e0cb-4777-a58d-cc6913552b45"):
    return helper.upload_file_process_send_s3(uploaded_file=uploaded_file,
                                              collection_id=collection_id)


@app.post("/del_doc_with_index_from_s3_collection")
def del_doc_with_index_from_s3_collection(collection_id: str = "378a73bc-e0cb-4777-a58d-cc6913552b45",
                                          file_name_with_extension: str = "4647bec2-3678-11ee-ae3b-00d49ea32059.pdf"):
    return helper.del_doc_with_index_from_s3_collection(collection_id=collection_id,
                                                        file_name_with_extension=file_name_with_extension)


@app.post("/del_s3_collection")
def del_s3_collection(collection_id: str = "378a73bc-e0cb-4777-a58d-cc6913552b45"):
    return helper.del_s3_collection(collection_id=collection_id)


@app.post("/del_local_collection")
def del_local_folder(local_collection_name: str = "378a73bc-e0cb-4777-a58d-cc6913552b45"):
    return helper.del_local_collection(local_collection_name=local_collection_name)


@app.post("/download_collection_from_s3_to_local")
def download_collection_from_s3_to_local(collection_id: str = "378a73bc-e0cb-4777-a58d-cc6913552b45"):
    return helper.download_collection_from_s3_to_local(collection_id=collection_id)


@app.post("/ask_to_llm_with_local_collection")
def ask_to_llm_with_local_collection(collection_id: str = "378a73bc-e0cb-4777-a58d-cc6913552b45",
                                     embedding_model_number: int = 0,
                                     top_K: int = 10,
                                     top_n: int = 3,
                                     llm: str = "openai",
                                     engine_name: str = 'gpt-3.5-turbo',
                                     temperature: float = 0.0,
                                     reduction_type: str = 'map_reduce',
                                     question: str = "Gece çalışması nedir?"):
    return helper.ask_to_llm_with_local_collection(collection_id=collection_id,
                                                   embedding_model_number=embedding_model_number,
                                                   top_K=top_K,
                                                   top_n=top_n,
                                                   llm=llm,
                                                   engine_name=engine_name,
                                                   temperature=temperature,
                                                   reduction_type=reduction_type,
                                                   question=question)
