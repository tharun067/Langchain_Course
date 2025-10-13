import streamlit as st
from langchain_groq import ChatGroq
from langchain.chains.llm import LLMChain
from langchain.chains.llm_math.base import LLMMathChain
from langchain.prompts import PromptTemplate
from langchain_community.utilities import WikipediaAPIWrapper
from langchain.agents.agent_types import AgentType
from langchain.agents import initialize_agent, Tool
from langchain.callbacks import StreamlitCallbackHandler
from dotenv import load_dotenv
load_dotenv()

### Set up streamlit app
st.set_page_config(page_title="Text to Math problem solver and Data search Assistant", page_icon=":robot:")
st.title("Text to Math problem solver Using Groq")


llm = ChatGroq(model="llama-3.3-70b-versatile")
## Intitalizeing the tools
wikipedia_wrapper = WikipediaAPIWrapper()
wikipedia_tool = Tool(
    name="Wikipedia",
    func=wikipedia_wrapper.run,
    description="Tool for searching up current information on Internet."
)

## Initializing the math tool
math_chain = LLMMathChain.from_llm(llm=llm)
calculator = Tool(
    name="Calculator",
    func=math_chain.run,
    description="Useful for when you need to answer questions about math. only mathematical calculations are allowed"
)

prompt = """
You are a agent tasked for solving math problems and searching for information on the internet.
Logically think through the problem and give a detailed response in the point wise format.
Question: {input}
Answer:
"""
prompt_template = PromptTemplate(
    input_variables=["input"],
    template=prompt
)

## Combine all the tools
chain = LLMChain(llm=llm, prompt=prompt_template)

resoning_tool = Tool(
    name="Reasoning Chain",
    func=chain.run,
    description="A Tool for answering logic and reasoning questions."
)

### Initialize the agent
assistant_agent = initialize_agent(
    tools=[wikipedia_tool, calculator, resoning_tool],
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=False,
    handle_parsing_errors=True
)

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role":"assistant","content":"Hi! I am a Math chatbot who can answer all your math problems."}
    ]

for message in st.session_state.messages:
    st.chat_message(message["role"]).write(message["content"])

### Function to generate response
def generate_response(user_question):
    response = assistant_agent.invoke({"input": user_question})
    return response

### Interaction
question = st.text_area("Enter your Question:","I have 5 bananas and 7 grapes. I eat 2 bananas and give 3 grapes to my friend. How many fruits do I have now?")
if st.button("Find my Answer"):
    if question:
        with st.spinner("Generating response..."):
            st.session_state.messages.append({"role":"user","content":question})
            st.chat_message("user").write(question)

            # Call the agent with the user's question string. .run returns the final text answer.
            st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
            try:
                # Use .run which accepts a plain string input and returns a textual response
                response_text = assistant_agent.run(question, callbacks=[st_cb])
            except Exception as e:
                # Fall back to invoke with the standard input key if .run fails
                response_obj = assistant_agent.invoke({"input": question}, callbacks=[st_cb])
                # Try to coerce to string for display
                response_text = str(response_obj)

            # Store and display the assistant reply as plain text
            st.session_state.messages.append({"role":"assistant","content":response_text})
            st.write("Response:")
            st.write(response_text)
    else:
        st.error("Please enter a question")

