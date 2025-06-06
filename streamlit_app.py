import streamlit as st
import requests
import subprocess
import atexit
import knowledge_converter as kc
import qna as qna
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI


# CRAWL4AI - FASTAPI - LAUNCH
# START
#def launch_fastapi():
    # Start FastAPI subprocess
#    return subprocess.Popen(["uvicorn", "crawler:app", "--host", "127.0.0.1", "--port", "8000"])

# Launch FastAPI only if not already running in this session
#if "api_process" not in st.session_state:
#    st.session_state.api_process = launch_fastapi()

# Register termination of the subprocess when Streamlit exits
#atexit.register(lambda: st.session_state.api_process.terminate())


# Wait a bit to ensure API is up
#time.sleep(2)
# END

#### SESSION START ##
# 
# DEFINE CRED
GEMINI_API_KEY = st.secrets["CREDENTIAL"]["GEMINI_API_KEY"]
CRAWL4AI_API_URL = st.secrets["CREDENTIAL"]["CRAWL4AI_URL"]



if "chats" not in st.session_state:
    st.session_state.chats = []

if "chats_llm" not in st.session_state:
    st.session_state.chats_llm = []

if "url" not in st.session_state:
    st.session_state.url = None

if "button_submitted" not in st.session_state:
    st.session_state.button_submitted = False

if "db" not in st.session_state:
    st.session_state.db = None

if "qna_chain" not in st.session_state:
    st.session_state.qna_chain = None

if "llm" not in st.session_state:


    # predefine llm here
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        temperature=0,
        max_tokens=None,
        timeout=None,
        max_retries=2,
        google_api_key=GEMINI_API_KEY
        # other params...
    )
    st.session_state.llm = llm



### STREAMLIT LAUNCH ##

st.set_page_config(page_title="Crawl4AI with Streamlit", layout="wide")
st.title("🌐 WebInsight AI")
st.write('')
st.divider()


# SIDEBAR
with st.sidebar:
    with st.form("my_form"):
        url = st.text_input("🌐 Enter website URL", placeholder="https://example.com")
        submitted = st.form_submit_button("Submit")
        if submitted:

            try:
                st.session_state.button_submitted = True
                st.session_state.url = url
                st.session_state.chats = []
                st.session_state.chats_llm = []
                st.session_state.qna_chain = None
                st.session_state.db = None


                response = requests.get(CRAWL4AI_API_URL, params={"target_url": url})
                data = response.json()

                if len(data['urls']) <= 0:
                    st.info('the web cannot be accessed or invalid url')
                else:

                    # Pass for knowledge converter
                    st.session_state.db = kc.convert_to_vector(data['result'], data['urls'])

                    retrieved = st.session_state.db.similarity_search("What is the name of the company?", k=3)

                    # uncomment for debug
                    # for r in retrieved:
                    #    print("✅ Match found:", r.page_content[:200])

                    # prep for qna
                    st.session_state.qna_chain = qna.qna_define(st.session_state.db, st.session_state.llm)
            except:
                st.warning('Error')





# below is for chat
prompt = st.chat_input('Ask anything...')


if prompt:
    
    if st.session_state.db != None:

        st.session_state.chats.append({
            "role": "user",
            "content": prompt
        })

        # invoke
        result = qna.qna_action(st.session_state.chats_llm, st.session_state.qna_chain, prompt)

        st.session_state.chats.append({
            "role": "assistant",
            "content": result
        })

    else:
        st.session_state.chats.append({
            "role": "user",
            "content": prompt
        })
    
        st.session_state.chats.append({
            "role": "assistant",
            "content": "Please input url"
        })



for chat in st.session_state.chats:
    st.chat_message(chat['role']).markdown(chat['content'])




