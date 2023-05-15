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

st.set_page_config(
   page_title="Frame Bot",
   page_icon=":robot_face:",
   initial_sidebar_state="collapsed",
   menu_items={
        'About': "This is a fighting game chat bot!"
    }
)

def read_markdown_file():
    path = Path('introduction.md')
    return path.read_text()

intro_markdown = read_markdown_file()

st.sidebar.markdown(intro_markdown, unsafe_allow_html=True)

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

    store = FAISS.from_texts(docs, OpenAIEmbeddings(openai_api_key= st.secrets["OPENAI_API_KEY"]), headers=["page_content"])
    faiss.write_index(store.index, "training.index")
    store.index = None

    with open("faiss.pkl", "wb") as f:
        store = pickle.dump(store, f)

index = faiss.read_index("training.index")

with open("faiss.pkl", "rb") as f:
    store = pickle.load(f)

store.index = index

# Create master prompt
masterPrompt = """You are a Professional Fighting game expert for Street Fighter 6 with years of experience teaching and explaining fighting games to new fighting game players. 
I want you to be a teach and explain things as if I had never played a fighting game before. You also have a strong understanding of frame data.

If any questions are asked that you don't know the answer to, please say "I don't know. Is there anything else I can help you with?" and move on to the next question.
If any questions are asked that aren't related to fighting games, please say "I don't know, please ask a question related to fighting games" and move on to the next question.

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

llmChain = LLMChain(prompt=prompt, llm=OpenAI(temperature=0, openai_api_key= st.secrets["OPENAI_API_KEY"]))

if 'generated' not in st.session_state:
    st.session_state['generated'] = []

if 'past' not in st.session_state:
    st.session_state['past'] = []

if "temp" not in st.session_state:
    st.session_state['temp'] = ''

def submit():
    st.session_state['temp'] = st.session_state['text']
    st.session_state['text'] = ''

history = []
st.markdown(":robot_face: Frame Bot")
st.text_input("Ask a question > ", placeholder="What is Drive Gauge?", key="text", on_change=submit)

question = st.session_state['temp']

# Create conversation subroutine (Do similarity search on the store then send to llmChain and come back with answer)
def onMessage(question, history):
    docs = store.similarity_search(question)
    contexts = []
    for i,doc in enumerate(docs):
        contexts.append(f"Context {i}:\n{doc.page_content}")
    answer = llmChain.predict(question=question, context="\n\n".join(contexts), history=history)
    return answer

if question:
    answer = onMessage(question, history)
    st.session_state.past.append(f"Human: {question}")
    st.session_state.generated.append(f"Bot: {answer}")
    history.append(f"Fighter: {question}")
    history.append(f"Frame Bot: {answer}")

if st.session_state['generated']:
    for i in range(len(st.session_state['generated'])):
        message(st.session_state['generated'][i])
        message(st.session_state['past'][i], is_user=True, avatar_style="adventurer", seed=123)
