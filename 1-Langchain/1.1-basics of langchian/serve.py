from fastapi import FastAPI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from langserve import add_routes
import os
from dotenv import load_dotenv
load_dotenv()

model = ChatGroq(model="Gemma2-9b-It")

# 1. Create a prompt template
generic_template = "Translate the following into {language}: "
prompt = ChatPromptTemplate.from_messages(
    [("system",generic_template),("human", "{text}")]
)
# 2. Create an output parser
parser = StrOutputParser()

# 3. Chain them together
chain = prompt | model | parser

## App definition

app = FastAPI(title="LangServe with Groq and LangChain Core", version="0.1", description="A simple API using LangServe, Groq LLM and LangChain Core")

## Adding chain routes
add_routes(
    app,
    chain,
    path="/chain"
)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8000)