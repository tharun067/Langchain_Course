import streamlit as st
from pathlib import Path
from langchain_community.agent_toolkits.sql.base import create_sql_agent
from langchain_community.utilities import SQLDatabase
from langchain.agents.agent_types import AgentType
from langchain_community.callbacks import StreamlitCallbackHandler
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
from sqlalchemy import create_engine
import sqlite3
from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()


st.set_page_config(page_title="ChatSQL", page_icon="🦜")
st.title("🦜LangChain: Chat with SQL DB")

LOCALDB = "USE_LOCALDB"
MYSQLDB = "USE_MYSQLDB"

radio_opt = ["Use Sqlite3 Database - Student.db", "Connect to MySQL Database"]

selected_opt = st.sidebar.radio(label="Select Database Option to chat", options=radio_opt)

if radio_opt.index(selected_opt) == 1:
    db_uri = MYSQLDB
    mysql_host = st.sidebar.text_input("Mysql Host")
    mysql_user = st.sidebar.text_input("Mysql User")
    mysql_password = st.sidebar.text_input("Mysql Password", type="password")
    mysql_db = st.sidebar.text_input("Mysql Database Name")
else:
    db_uri = LOCALDB

if not db_uri:
    st.info("Please select a database option to proceed.")


### LLM model


llm = ChatGroq(model="llama-3.3-70b-versatile", streaming=True)

@st.cache_resource(ttl="2h")
def configure_db(db_uri, mysql_host=None, mysql_user=None, mysql_password=None, mysql_db=None):
    if db_uri == LOCALDB:
        db_path = (Path(__file__).parent / "Student.db").absolute()
        print(f"Loading local db from {db_path}")
        creator = lambda: sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        return SQLDatabase(create_engine("sqlite:///", creator=creator))
    elif db_uri == MYSQLDB:
        if not (mysql_db and mysql_host and mysql_user and mysql_password):
            st.error("Please enter all MySQL connection details.")
            st.stop()
        return SQLDatabase(create_engine(f"mysql+mysqlconnector://{mysql_user}:{mysql_password}@{mysql_host}/{mysql_db}"))

if db_uri == MYSQLDB:
    db = configure_db(db_uri, mysql_host, mysql_user, mysql_password, mysql_db)
else:
    db = configure_db(db_uri)

### toolkit
toolkit = SQLDatabaseToolkit(db=db, llm=llm)

agent = create_sql_agent(llm=llm, toolkit=toolkit, verbose=True, agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION)

if "messages" not in st.session_state or st.sidebar.button("Clear Conversation"):
    st.session_state["messages"] = [{"role":"assistant", "content":"How can I help you?"}]

for msg in st.session_state["messages"]:
    st.chat_message(msg["role"]).write(msg["content"])

user_query = st.chat_input(placeholder="Ask anything about the database...")

if user_query:
    st.session_state["messages"].append({"role":"user", "content":user_query})
    st.chat_message("user").write(user_query)

    with st.chat_message("assistant"):
        streamlit_callback = StreamlitCallbackHandler(st.container())
        response = agent.run(user_query, callbacks=[streamlit_callback])
        st.session_state["messages"].append({"role":"assistant", "content":response})
        st.write(response)