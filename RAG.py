import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain_community.document_loaders import WebBaseLoader
from langchain.embeddings import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
import time
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders.csv_loader import CSVLoader




llm=ChatGroq(groq_api_key="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",
             model_name="mixtral-8x7b-32768")

embeddings=OllamaEmbeddings()


def  web_loader():
    with st.spinner("Please wait for few Seconds.Almost there"):
        prompte=ChatPromptTemplate.from_template(
"""
Answer the questions based on the provided context only.
Please provide the most accurate response based on the question
<context>
{context}
<context>
Questions:{input}

"""
)
        loader=WebBaseLoader("https://www.jagranjosh.com/general-knowledge/ms-dhoni-biography-1594085953-1")
        docs=loader.load()
        text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
        final_documents=text_splitter.split_documents(docs)
        vectors=FAISS.from_documents(final_documents,embeddings)
        document_chain = create_stuff_documents_chain(llm, prompte)
        retriever = vectors.as_retriever()
        retrieval_chain = create_retrieval_chain(retriever, document_chain)
        prompt=input_prompt
        response=retrieval_chain.invoke({"input":prompt})
    return response['answer']

def pdf_loader():
    with st.spinner("Please wait for few Seconds.Almost there"):
        prompte=ChatPromptTemplate.from_template(
"""
Answer the questions based on the provided context only.
Please provide the most accurate response based on the question
<context>
{context}
<context>
Questions:{input}

"""
)
        loader1 = PyPDFLoader("Free Fresher Resume Sample.pdf")
        docs1=loader1.load()
        text_splitter1=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
        final_documents1=text_splitter1.split_documents(docs1)
        vectors1=FAISS.from_documents(final_documents1,embeddings)
        document_chain = create_stuff_documents_chain(llm, prompte)
        retriever1 = vectors1.as_retriever()
        retrieval_chain1 = create_retrieval_chain(retriever1, document_chain)
        prompt1=input_prompt
        response1=retrieval_chain1.invoke({"input":prompt1})
    return response1['answer']

def csv_loader():
    with st.spinner("Please wait for few Seconds.Almost there"):
        prompte=ChatPromptTemplate.from_template(
"""
Answer the questions based on the provided context only.
Please provide the most accurate response based on the question
<context>
{context}
<context>
Questions:{input}

"""
)
        loader2 = CSVLoader("people.csv")
        docs2=loader2.load()
        text_splitter2=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
        final_documents2=text_splitter2.split_documents(docs2)
        vectors2=FAISS.from_documents(final_documents2,embeddings)
        document_chain = create_stuff_documents_chain(llm, prompte)
        retriever2 = vectors2.as_retriever()
        retrieval_chain2 = create_retrieval_chain(retriever2, document_chain)
        prompt2=input_prompt
        response2=retrieval_chain2.invoke({"input":prompt2})
    return response2['answer']


input_prompt=st.text_input("Enter Your Question From Documents")

if st.button("website"):
    
    st.write(web_loader())# website  about ms dhoni
    

if st.button("PDF"):
    
    st.write(pdf_loader())# resume of guy named Saurabh Gupta.
    

if st.button("csv"):
    
    st.write(csv_loader()) #  dataset people.csv containg basic info like gender,email,phone number etc. 
    

