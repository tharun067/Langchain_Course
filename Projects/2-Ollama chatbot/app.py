from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.llms import ollama
from posthog import api_key
import streamlit as st

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

def generate_response(question, engine, temperature):
    llm = ollama.Ollama(model=engine,temperature=temperature)
    output_parser = StrOutputParser()
    chain = prompt | llm | output_parser
    answer = chain.invoke({'question':question})
    return answer

# Title of the app
st.title("Enhanced Q&A Chatbot with Ollama")

# Drop down to select Ollama Models
llm = st.sidebar.selectbox("Select an Ollama AI Model",["gemma2:2b","DeepSeek-R1","Mistral"])

## Adjust response parameters
temperature = st.sidebar.slider("Temperature",min_value=0.0,max_value=1.0,value=0.7)


## Main interface for user input
st.write("Go head and ask me anything!")
user_input = st.text_area("Your Question:",height=70)

if user_input:
    response = generate_response(user_input,llm,temperature)
    st.write(response)
else:
    st.write("Please enter a question to get a response.")