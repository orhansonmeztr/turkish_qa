import helper
from fastapi import FastAPI, UploadFile
import uvicorn

app = FastAPI(title="Endpoints for QA over Documents",
              description="For now you can only use 2 and 4 as embedding_model_number")


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


@app.post("/ask_to_llm_with_local_collection")
def ask_to_llm_with_local_collection(collection_id: str = "378a73bc-e0cb-4777-a58d-cc6913552b45",
                                     embed_model_number: int = 2,
                                     top_k: int = 10,
                                     top_n: int = 3,
                                     llm: str = "openai",
                                     engine_name: str = 'gpt-3.5-turbo',
                                     llm_temp: float = 0.0,
                                     reduction_type: str = 'map_reduce',
                                     question: str = "Gece çalışması nedir?"):
    return helper.ask_to_llm_with_local_collection(collection_id=collection_id,
                                                   embed_model_number=embed_model_number,
                                                   top_k=top_k,
                                                   top_n=top_n,
                                                   llm=llm,
                                                   engine_name=engine_name,
                                                   llm_temp=llm_temp,
                                                   reduction_type=reduction_type,
                                                   question=question)

# For debugging
# if __name__ == '__main__':
#     uvicorn.run(app, host='0.0.0.0', port=8000)
