import glob, os
import streamlit as st
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence, RunnableWithMessageHistory, RunnablePassthrough
from streamlit_chat import message
from dotenv import load_dotenv, find_dotenv
from langchain.chains.llm import LLMChain
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.prompts import PromptTemplate
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain_community.chat_message_histories.streamlit import (
    StreamlitChatMessageHistory,
)
from langchain.globals import get_verbose

get_verbose()

load_dotenv(find_dotenv(), override=True)

context = "\n".join(
    open(file_path, "r", encoding="utf-8").read()
    for file_path in glob.glob(os.path.join("data", "*.txt"))
)

template = """You are an empathetic, kind and helpful AI Human Resource assistant having a conversation with a human. 
You are able to handle both english and german as input and output. Always answer the question in the language of the question. 
Restrict your answers to the information provided in the context only do not make up any facts yourself. 
If you do not know the answer, say that you do not know and the user should contact HR directly under the following email hr@rewe-group.com.

{context}

{history}
Human: {human_input}
AI: """
prompt = PromptTemplate(input_variables=["context", "history", "human_input"], template=template)

msgs = StreamlitChatMessageHistory(key="special_app_key")

memory = ConversationBufferMemory(memory_key="history", chat_memory=msgs)

def load_chain():
    llm = ChatGoogleGenerativeAI(model="models/gemini-1.5-pro-002", api_key="AIzaSyBxbOxq1tO-x6rbw0xWfOjO1APlybjBPBc")
    chain = (
        {"context": lambda x: context, "human_input": RunnablePassthrough()}
        | RunnablePassthrough.assign(
            history=lambda x: memory.load_memory_variables(x)["history"]
        )
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain


def initialize_session_state():
    if "chain" not in st.session_state:
        st.session_state.chain = load_chain()

    if "generated" not in st.session_state:
        st.session_state.generated = []

    if "past" not in st.session_state:
        st.session_state.past = []

    if "user_input" not in st.session_state:
        st.session_state.user_input = ""

    if "widget_input" not in st.session_state:
        st.session_state.widget_input = ""


initialize_session_state()

st.set_page_config(page_title="Easy HR ChatBot", page_icon=":robot:")
st.header("Easy HR ChatBot")


def submit():
    st.session_state.user_input = st.session_state.widget_input
    st.session_state.widget_input = ""


st.text_input("You:", key="widget_input", on_change=submit)

if st.session_state.user_input:
    input_dict = {"human_input": st.session_state.user_input}
    output = st.session_state.chain.invoke(input_dict)
    memory.save_context(
        {"human_input": st.session_state.user_input},
        {"output": output}
    )
    st.session_state.past.append(st.session_state.user_input)
    st.session_state.generated.append(output)
    st.session_state.user_input = ""

if st.session_state["generated"]:
    for i in range(len(st.session_state["generated"]) - 1, -1, -1):
        message(st.session_state["generated"][i], key=str(i))
        message(st.session_state["past"][i], is_user=True, key=str(i) + "_user")
