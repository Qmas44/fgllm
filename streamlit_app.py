import os, sys
from pathlib import Path

import streamlit as st
from langchain.text_splitter import CharacterTextSplitter
import faiss
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
import pickle

if "OPENAI_API_KEY" not in os.environ:
  print("You must set an OPENAI_API_KEY using the Secrets tool", file=sys.stderr)
  exit()

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

with open("training.store", "wb") as f:
    store = pickle.dump(store, f)


st.write('helloooo')
