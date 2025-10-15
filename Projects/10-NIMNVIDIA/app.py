import streamlit as st
import os
import time
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings, ChatNVIDIA
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS

from dotenv import load_dotenv

load_dotenv()

os.environ["NVIDIA_API_KEY"] = os.getenv("NVIDIA_API_KEY")

llm = ChatNVIDIA(model="meta/llama3-70b-instruct", temperature=0)

def vector_embeddings():
    if "vectors" not in st.session_state:
        st.session_state.embeddings = NVIDIAEmbeddings()
        st.session_state.loader = PyPDFDirectoryLoader("./us_census")
        st.session_state.docs = st.session_state.loader.load()
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=50)
        st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs[:30])
        st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)


st.title("NIM with NVIDIA Embeddings and LLMs")

prompt = ChatPromptTemplate.from_template(
    """
Answer the questions based on the provided context only.
Please provide most accurate response based on the question.
<context>
{context}
</context>
Question: {input}
"""
)

prompt1 = st.text_input("Enter your question here", "What is the population of the US?")

if st.button("Document Embeddings"):
    vector_embeddings()
    st.success("FAISS Vector Store created and stored in session state using NVIDIA Embeddings!")

if prompt1:
    document_chain = create_stuff_documents_chain(llm=llm, prompt=prompt)
    retriever = st.session_state.vectors.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    start = time.process_time()
    response = retrieval_chain.invoke({'input':prompt1})
    st.write(f"Response time: {time.process_time() - start} seconds")
    st.write(response['answer'])


    ## With stramlit expander
    with st.expander("Document Similarity Search Results"):
        for i, doc in enumerate(response['context']):
            st.write(doc.page_content)
            st.write("----------------------------------")