import helper
from fastapi import FastAPI, UploadFile
from fastapi.responses import RedirectResponse

app = FastAPI()


@app.get('/api/hc')
def health_check():
    return {'status': 'ok'}


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.post("/upload_process_send_s3")
def upload_process_send_s3(uploaded_file: UploadFile,
                           collection_id: str = "378a73bc-e0cb-4777-a58d-cc6913552b45",
                           embedding_model_number: int = 0):
    return helper.upload_process_send_s3(uploaded_file=uploaded_file,
                                         collection_id=collection_id,
                                         embedding_model_number=embedding_model_number)


# @app.post("/del_local_file")
# def del_local_file_folder(local_file_name: str = "7234163a-311a-11ee-a1b4-00d49ea32059.pdf"):
#     return helper.del_local_file(local_file_name=local_file_name)
#
#
# @app.post("/del_local_folder")
# def del_local_folder(local_folder_name: str = "new-folder"):
#     return helper.del_local_folder(local_folder_name=local_folder_name)
#
#
# @app.post("/process_pdf_send_s3")
# def process_pdf_send_s3(local_file: str = "e04baa17-3200-11ee-a27d-00d49ea32059.pdf",
#                         collection_id: str = "378a73bc-e0cb-4777-a58d-cc6913552b45",
#                         embedding_model_number: int = 0):
#     return helper.process_pdf_send_s3(local_file=local_file,
#                                       collection_id=collection_id,
#                                       embedding_model_number=embedding_model_number)


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
