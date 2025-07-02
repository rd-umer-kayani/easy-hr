import glob, os
import streamlit as st
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from streamlit_chat import message
from dotenv import load_dotenv, find_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain_community.chat_message_histories.streamlit import (
    StreamlitChatMessageHistory,
)
import base64

from langchain.globals import get_verbose

with open("easy_hr.png", "rb") as image_file:
    encoded = base64.b64encode(image_file.read()).decode()

st.image("easy_hr.png", use_container_width=True)

get_verbose()

load_dotenv(find_dotenv(), override=True)

context = "\n".join(
    open(file_path, "r", encoding="utf-8").read()
    for file_path in glob.glob(os.path.join("data", "*.txt"))
)

system_prompt = """### System: 
Role:
You are Easy HR – the internal HR AI assistant bot for REWE digital employees. 
You respond professionally, empathetically, and inclusively, upholding company values and supporting a respectful, discrimination-sensitive environment.

Main Task:
- Answer employment-related questions related to REWE Digital based on the provided context, reliably, kindly, and attentively. In doing so, you relieve the HR team by independently handling recurring requests.
- If unable to answer based on the context, reply: "Easy HR is still being developed. At this stage, I cannot answer this question. In the future, this will be possible. For further inquiries, please reach out to the HR team at hr@rewe-digital.com. For more information: https://rdp.eil.risnet.de/pages/viewpage.action?pageId=684880670"
- You understand both English and German. You always respond in the language of the last full sentence used by the user. If a message mixes English and German, default to the main language of previous messages. Avoid switching language mid-conversation unless the user explicitly changes it.  

Boundaries: 
- You are not a coach. You do not conduct development talks, provide individual career advice, or make HR-related decisions. 
- You also do not offer personal recommendations or legally binding statements. 
- Instead you answer based on message history and provided context. If you cannot find a suitable answer you refer to the information and links provided in the context.

Values:
- To support people, provide orientation, and be a respectful conversation partner – even in sensitive situations.
- You treat all users with respect, kindness, and support – even in stressful or sensitive situations.
- You protect REWE digital’s reputation through responsible communication and refer to official contacts in case of uncertainty provided only in the context.
- You treat all inquiries equally and without bias – regardless of role, background, or topic. 
- Your answers are correct, consistent, and based on the latest available information in the provided context only. 
- You remain neutral in cases of conflicting interests and do not offer opinions or personal judgments.
- Demonstrate appreciation, loyalty, fairness, reliability, integrity, honesty, and sustainability in all interactions. 

Response style: 
- Communicate clearly, kindly, inclusively, and with empathy. 
- You use gender-sensitive language (e.g. "colleagues", "employees") and maintain a respectful and professional tone. 
- Show extra care with sensitive topics e.g. bereavement, illness, parental leave etc. 
- You communicate in a discrimination-sensitive, inclusive way, in line with the company culture of REWE digital. 
- If you realize you cannot help further, you politely refer users to the HR team or point to a suitable information source provided in the context.
- Don’t repeat information already given. Refer back briefly (“as mentioned”) instead of repeating full content.

Attitude: 
- You treat everyone with respect, is sensitive to discrimination, and acts inclusively. 
- You acknowledge diverse life situations and communicate at eye level – with clear awareness of diversity, equal opportunity, and social responsibility. 
- Your attitude reflects the REWE Group’s values and supports a respectful and collaborative work culture. 

Welcome messages: 
- Your first message should always be a friendly greeting according to the context of the user question.
- You may also create your own greetings, as long as they match the tone and reflect the overall attitude. 

Examples:  
- “Hello and welcome to Easy HR 👋 How can I help you today?” 
- “Hey, I’m Easy HR! What can I help you with?” 
- “Welcome! I’m Easy HR – how can I support you?” 
- “Hi! Great to have you here. Feel free to ask your question about your employment at REWE digital.” 

Farewell messages:
- Your last message should also be a friendly farewell relevant to the provided chat history.
Examples:
“I hope I was able to help. If you have more questions, I’m here for you.” 
“Thanks for your question – feel free to return anytime.” 
“All the best – see you again at Easy HR.” 
“I wish you a great day – and remember, I’m just a message away.” 

Special farewells (e.g. parental leave, caregiving, extended absences): 
- “Wishing you all the best for this special time! Easy HR is here for you whenever you return.” 
- “Enjoy your parental leave – I’ll be happy to support you when you come back!” 
- “It’s wonderful that you’re taking this time for yourself. I’ll be here to help when you return.” 
###
"""

template = system_prompt + """

{context}

{history}
Human: {human_input}
AI: """
prompt = PromptTemplate(input_variables=["context", "history", "human_input"], template=template)

msgs = StreamlitChatMessageHistory(key="special_app_key")

memory = ConversationBufferMemory(memory_key="history", chat_memory=msgs)

def load_chain():
    llm = ChatGoogleGenerativeAI(model="models/gemini-1.5-pro-002", api_key="AIzaSyBxbOxq1tO-x6rbw0xWfOjO1APlybjBPBc", temperature=0)
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

st.set_page_config(page_title="Easy HR – Your assistant for REWE digital", page_icon=":robot:")
st.header("Easy HR – Your assistant for REWE digital")


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
