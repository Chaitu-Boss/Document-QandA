import streamlit as st
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_community.embeddings import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.document_loaders import PyPDFLoader
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.vectorstores import FAISS

load_dotenv()

llm=ChatGroq(model="llama3-8b-8192",api_key=os.getenv("GROQ_API_KEY"))
prompt=ChatPromptTemplate.from_template(
   """
    You are a helpful assistant who answers question asked by user from his document
    You are also given a context of document
    {context}
    Question: {input}
    """
    
)
def create_vector_embedding(uploaded_pdf):
    try:
        if "vectors" not in st.session_state:
            with open("temp_uploaded_file.pdf", "wb") as f:
                f.write(uploaded_pdf.getbuffer())
            st.session_state.embeddings=OllamaEmbeddings(model='nomic-embed-text')
            loader=PyPDFLoader("temp_uploaded_file.pdf") 
            st.session_state.docs=loader.load()
            if not st.session_state.docs:
                st.error("No content found in the uploaded PDF.")
                return
            text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
            st.session_state.final_documents=text_splitter.split_documents(st.session_state.docs)
            if not st.session_state.final_documents:
                st.error("Not enought content found in the PDF.")
                return
            st.session_state.vectors = FAISS.from_documents(
                st.session_state.final_documents, st.session_state.embeddings
            )
            st.success("Vector database created successfully!")
    except Exception as e:
        st.error(f"Internal Error: {e}")
    
st.title("RAG Document Q&A With Groq And Ollama")

uploaded_pdf=st.file_uploader("Upload your PDF here",type="pdf")
if uploaded_pdf is not None:
    if st.button("Process PDF and Create Vector Database"):
        create_vector_embedding(uploaded_pdf)

user_prompt=st.text_input("Enter your question here")

import time

if user_prompt and "vectors" in st.session_state:
    document_chain=create_stuff_documents_chain(llm,prompt)
    retriever=st.session_state.vectors.as_retriever()
    retriever_chain=create_retrieval_chain(retriever,document_chain)
    response=retriever_chain.invoke({'input':user_prompt})
    start=time.process_time()
    st.write(response['answer'])
    st.write(f"Time Taken : {time.process_time()-start} seconds")

temp_file = "temp_uploaded_file.pdf"
if os.path.exists(temp_file):
    os.remove(temp_file)
    print(f"Temporary file {temp_file} has been removed.")
else:
    print(f"Temporary file {temp_file} does not exist.")


    
