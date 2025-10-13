import validators
import streamlit as st
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain.chains.summarize import load_summarize_chain
from langchain_community.document_loaders import YoutubeLoader, UnstructuredURLLoader

## Streamlit app
st.set_page_config(page_title="Text Summarization bot",page_icon="🦜")
st.title("🦜 Langchain: Summarize Text from YT or Webiste")
st.subheader("Summarize URL")

## Get the Groq API key and url to summarize
with st.sidebar:
    groq_api_key = st.text_input("Groq API Key",value="",type="password")

url = st.text_input("URL",label_visibility="collapsed")
if groq_api_key:
    llm = ChatGroq(model="llama-3.3-70b-versatile",api_key=groq_api_key)

## 
prompt_template = """
Provide a summary of the following content in 300 words:
Content: {text}
"""

prompt = PromptTemplate(input_variables=['text'],template=prompt_template)


if st.button("Summarize the Content from YT or Website"):
    ### Vaildate all inputs
    if not groq_api_key.strip() or not url.strip():
        st.error("Please provide the information")
    elif not validators.url(url):
        st.error("Please enter a vaild Url. It can be Youtube Url or any Website url..")
    else:
        try:
            with st.spinner("Waiting.."):
                ### Load the data
                if "youtube.com" in url:
                    loader = YoutubeLoader.from_youtube_url(url,add_video_info=True)
                else:
                    loader = UnstructuredURLLoader(urls=[url],ssl_verify=False,headers={"User-Agent":"Mozilla/5.0 (Macintosh; Intel Mac OS X 13_5_1) AppleWebKit/537.36 (KHTML, like Gecko) Chorme/116.0.0.0 Safari/537.36"})

                docs = loader.load()

                ##Chain for summarization
                chain = load_summarize_chain(llm, chain_type="stuff",prompt=prompt)
                output_summary = chain.run(docs)

                st.success(output_summary)

        except Exception as e:
            st.exception(f"Exception: {e}")