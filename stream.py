import helper
import streamlit as st

st.title("Ask question about basın_is_kanunu.pdf and deniz_is_kanunu.pdf")
question = st.text_input("Your question:")
if question:
    result = helper.ask_to_llm_with_local_collection(collection_id="378a73bc-e0cb-4777-a58d-cc6913552b45",
                                                     embedding_model_number=0,
                                                     top_K=10,
                                                     top_n=3,
                                                     llm="openai",
                                                     engine_name='gpt-3.5-turbo',
                                                     temperature=0.0,
                                                     reduction_type='map_reduce',
                                                     question=question)
    st.write(result)
