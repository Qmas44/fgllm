import os, sys
from pathlib import Path
import streamlit as st
from langchain.text_splitter import CharacterTextSplitter
import faiss
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain import OpenAI, LLMChain
from langchain.prompts import PromptTemplate
import pickle

if "OPENAI_API_KEY" not in os.environ:
  print("You must set an OPENAI_API_KEY using the Secrets tool", file=sys.stderr)
  exit()

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

    store = FAISS.from_texts(docs, OpenAIEmbeddings())
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

llmChain = LLMChain(prompt=prompt, llm=OpenAI(temperature=0))

# Create conversation subroutine (Do similarity search in the docs then send to llmChain and come back with answer)
def onMessage(question, history):
    docs = store.similarity_search(question)
    contexts = []
    for i,doc in enumerate(docs):
        contexts.append(f"Context {i}:\n{doc.page_content}")
    answer = llmChain.predict(question=question, context="\n\n".join(contexts), history=history)
    return answer

history = []
while True:
    question = input("Ask a question > ")
    answer = onMessage(question, history)
    print(f"Bot: {answer}")
    history.append(f"Human: {question}")
    history.append(f"Bot: {answer}")


st.write('helloooo')
