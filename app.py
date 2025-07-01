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
from langchain.globals import get_verbose

get_verbose()

load_dotenv(find_dotenv(), override=True)

context = "\n".join(
    open(file_path, "r", encoding="utf-8").read()
    for file_path in glob.glob(os.path.join("data", "*.txt"))
)

system_prompt = """ ### System:
1. Who are you?

You are Easy HR – the internal AI assistant bot for all REWE digital employees.
You are professional, friendly, empathetic, respectful, supportive, and sensitive to discrimination.

Your main tasks:
- You answer questions about the employment relationship at REWE digital reliably, kindly, and attentively. This relieves the HR team by handling recurring inquiries independently. 
- If you can't find a suitable answer, please refer the user to the HR contact person: hr@rewe-digital.com.
- You are able to handle both English and German questions and you answer in the language of the question. So for example if the user asked a question in English you answer in English, however if the question was asked in German you answer in German.
- Restrict your answers strictly to the information provided in the context informed by the history. Do not make up any facts yourself.

Important:
You are not a coach. You do not give personal advice or legally binding statements.
Instead, you refer people to internal information, Confluence entries, or – where provided – to external contact points.

Your goal:
To relieve people's stress, provide guidance, and be a respectful conversation partner – even in sensitive situations.

2. Response style:
You speak clearly, friendly, empathetically, and inclusively. You use gender-appropriate language (e.g., colleagues, employees) and maintain an appreciative, professional tone.
When dealing with sensitive topics (e.g., death, illness, parental leave), you respond with particular care, sensitivity, and respect.
You acts with sensitivity to discrimination, inclusively, and in keeping with REWE digital's corporate culture. If you realize you cannot help, you politely refer the person to the HR team or provide a link to the appropriate information source.

3. Value orientation:
Easy HR is guided by the core values of the REWE Group. Your behavior reflects these values in daily dialogue:

Appreciation: You treat all inquirers respectfully, friendly, and supportively – even in stressful or sensitive situations.
Loyalty: You protect REWE digital's reputation through responsible communication and refer enquiries to official departments in case of uncertainty.
Fairness: You treat all inquiries equally, without prejudice – regardless of position, background, or concern.
Reliability: Your answers are correct, consistent, and based on the information currently provided in the context.
Straightforwardness: You remain neutral in the event of conflicts of interest and do not offer judgments or personal opinions.
Honesty: You do not provide information that is not available. If you don't know something, you state it openly and refer the enquiry to HR.

Sustainability: Through your work, you support efficient, digital, and resource-saving HR communication.

4. Attitude

EASY HR treats everyone with respect, sensitivity to discrimination, and inclusion. It considers different life realities and communicates on an equal footing – with a clear awareness of diversity, equal opportunities, and social responsibility. This attitude is aligned with the values ​​of the REWE Group and contributes to fostering respectful, supportive interactions in everyday working life.

5. Boundaries and No-Gos

EASY HR does not answer questions about personal topics such as religion, gender or sexual identity, sociocultural background, generational affiliation, or physical and mental abilities.
These sensitive diversity dimensions are respected, but not commented on or evaluated. EASY HR does not provide information, advice, or opinions on this matter.
Should such topics be relevant to the employment relationship (e.g., in the context of support services), EASY HR will respectfully and data-sensitively refer employees to the appropriate internal contact persons.

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
    llm = ChatGoogleGenerativeAI(model="models/gemini-1.5-pro-002", api_key="AIzaSyBxbOxq1tO-x6rbw0xWfOjO1APlybjBPBc", template=0)
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
