import streamlit as st
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

import os
from dotenv import load_dotenv

load_dotenv()

## Langsmith tracking
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "Q&A Chatbot"

## Prompt Template

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant. Please respond to the user's question as best as you can."),
        ("user", "Question: {question}"),
    ]
)

def generate_response(question, api_key, llm, temperature, max_tokens):
    llm = ChatGroq(model=llm, api_key=api_key,temperature=temperature,max_tokens=max_tokens)
    output_parser = StrOutputParser()
    chain = prompt | llm | output_parser
    answer = chain.invoke({'question':question})
    return answer

## Title of the app
st.title("Enhanced Q&A Chatbot with Groq")
st.sidebar.title("Settings")
api_key = st.sidebar.text_input("Enter your Groq API key: ", type="password")

## Drop down to select Groq Models
llm = st.sidebar.selectbox("Select an Groq AI Model",["llama-3.1-8b-instant","llama-3.3-70b-versatile","meta-llama/llama-guard-4-12b"])

## Adjust response parameters
temperature = st.sidebar.slider("Temperature",min_value=0.0,max_value=1.0,value=0.7)
max_tokens = st.sidebar.slider("Max Tokens",min_value=50,max_value=300,value=150)


## Main interface for user input
st.write("Go head and ask me anything!")
user_input = st.text_area("Your Question:",height=70)

if user_input:
    response = generate_response(user_input,api_key,llm,temperature,max_tokens)
    st.write(response)
else:
    st.write("Please enter a question to get a response.")