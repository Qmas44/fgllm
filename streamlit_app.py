import os, sys
from pathlib import Path

import streamlit as st

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

st.write('helloooo')
