import os, sys
from pathlib import Path
import langchain
from langchain.text_splitter import CharacterTextSplitter
import faiss
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain import OpenAI, LLMChain
from langchain.prompts import PromptTemplate
import pickle
import streamlit as st
from streamlit_chat import message
import streamlit as st
import streamlit.components.v1 as components

def read_file(file):
    path = Path(file)
    return path.read_text()

st.set_page_config(
   page_title="Frame Bot",
   page_icon=":robot_face:",
   initial_sidebar_state="auto",
   menu_items={
        'About': "This is a fighting game chat bot!"
    }
)

def hide_anchor_link():
    st.markdown("""
        <style>
        .css-15zrgzn {display: none}
        .css-eczf16 {display: none}
        .css-jn99sy {display: none}
        </style>
        """, unsafe_allow_html=True)
hide_anchor_link()

intro_markdown = read_file('introduction.md')

chat_box = read_file('chat_box.md')

st.sidebar.markdown(intro_markdown, unsafe_allow_html=True)

log = open("log.txt", "a")

# Pull training data from the training folder and store in vector store with caching
def train():
    trainingData = list(Path('training/').rglob("**/*.*"))
    print(len(trainingData))

    if len(trainingData) < 1:
        print("No files in the training folder", file=sys.stderr)
        exit()
    data = []
    for training in trainingData:
        with open(training) as f:
            print(f"Add {f.name} to dataset")
            data.append(f.read())

    textSplitter = CharacterTextSplitter(chunk_size=2000, separator="\n")

    docs = []
    for sets in data:
        docs.extend(textSplitter.split_text(sets))

    # Hugging face drawing errors sometimes. Need to find a better way to embed
    store = FAISS.from_texts(docs, OpenAIEmbeddings(openai_api_key=st.secrets["OPENAI_API_KEY"]))
    faiss.write_index(store.index, "training.index")
    store.index = None

    with open("faiss.pkl", "wb") as f:
        store = pickle.dump(store, f)
# make sure train() is called here
train()

index = faiss.read_index("training.index")

with open("faiss.pkl", "rb") as f:
    store = pickle.load(f)

store.index = index

# Create master prompt
masterPrompt = """You are a Professional Fighting game expert for Street Fighter 6 with years of experience teaching and explaining fighting games to new fighting game players. 
I want you to be a teach and explain things as if I had never played a fighting game before. You also have a strong understanding of frame data.

For frame data questions, do not use any other information except for the txt files provided in the training folder as outside information could be from previous games.

If any questions are asked that you don't know the answer to, please say "Sorry, I'm not sure. Is there anything else I can help you with?" and move on to the next question.
If any questions are asked that aren't related to Street Fighter or fighting games, please say "Sorry, I'm not sure. please ask a question related to Street Fighter" and move on to the next question.
Make sure to check the list of characters playable in Street fighter 6 before answering any questions. If a question is asked about a character that isn't playable in Street Fighter 6, please say "Sorry, I'm not sure. This character isn't playable in Street Fighter 6. Is there anything else I can help you with?" and move on to the next question.
Try to stay away from mentioning Street Fighter 5 systems like v-skill. Be sure to mention that Street Fighter 6 has a new Drive system.
If a special move is mentioned and the type (light, medium, or heavy) is not specified, please give the data for all versions of that special move. For example, if someone asks about Ryu's Hadoken, please give that information for the light, medium, and heavy version of the hadoken.

If there is a follow up question use the previous answer as the context for the next question.
Use the following pieces of MemoryContext to answer the questions at the end. Also remember ConversationHistory is a list of Conversation objects.
---
ConversationHistory: {history}
---
MemoryContext: {context}
---
Human: {question}
BOT:"""

# setup prompt to expect to see history, embeddings and question
prompt = PromptTemplate(template=masterPrompt, input_variables=["history", "context", "question"])

#llmChain = LLMChain(prompt=prompt, llm=OpenAI(temperature=0, openai_api_key= st.secrets["OPENAI_API_KEY"]))

if 'streamlit_chat' not in st.session_state:
    st.session_state['streamlit_chat'] = []

if 'generated' not in st.session_state:
    st.session_state['generated'] = []

if 'past' not in st.session_state:
    st.session_state['past'] = []

if "temp" not in st.session_state:
    st.session_state['temp'] = ''

st.markdown(chat_box, unsafe_allow_html=True)

# Create conversation subroutine (Do similarity search on the store then send to llmChain and come back with answer)
def onMessage(question, history):
    docs = store.similarity_search(question)
    contexts = []
    for i,doc in enumerate(docs):
        contexts.append(f"Context {i}:\n{doc.page_content}")
    answer = llmChain.predict(question=question, context="\n\n".join(contexts), history=history)
    return answer

history = []

def submit():

    st.session_state['temp'] = st.session_state['text']
    st.session_state['text'] = ''

    question = st.session_state['temp']

    answer = 'test' #onMessage(question, history)

    #log.write(f"Fighter: {question} ")
    #log.write(f"Frame Bot: {answer}\n")
    #log.close()

    history.append(f"Fighter: {question}")
    history.append(f"Frame Bot: {answer}")
    
    st.session_state['past'].append(question)
    st.session_state['generated'].append(answer)

# If responses have been generated by the model
if st.session_state['generated']:
    # Reverse iteration through the list
    for i in range(len(st.session_state['generated'])-1, -1, -1):
        # message from streamlit_chat
        message(st.session_state['past'][::-1][i], is_user=True, key=str(i) + '_user', avatar_style="adventurer", seed=123)
        message(st.session_state['generated'][::-1][i], key=str(i))

st.text_input("Ask a question > ", max_chars=150, placeholder="What is Drive Gauge?", key="text", on_change=submit)
