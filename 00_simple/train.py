import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import os

from model import NerualNet
from utils import tokenize, bag_of_words

if os.name == "nt":
    INTENTS_FILE = "00_simple\\intents.json"
else:
    INTENTS_FILE = "intents.json"

with open("00_simple\\intents.json", "r") as f:
    intents = json.load(f)

all_words = []
tags = []
xy = []

for intent in intents["intents"]:
    tag = intent["tag"]
    tags.append(tag)
    for pattern in intent["patterns"]:
        w = tokenize(pattern)
        all_words.extend(w)
        xy.append((w, tag))

ignore = ["?", "!", ".", ","]
all_words = sorted(set([w.lemma_ for w in all_words if w.orth_ not in ignore]))
tags = sorted(set(tags))

X_train = []
y_train = []

for (pattern, tag) in xy:
    bag = bag_of_words(pattern, all_words)
    X_train.append(bag)

    label = tags.index(tag)
    y_train.append(label)

X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)

dataset = list(zip(X_train, y_train))
loader = DataLoader(dataset=dataset, batch_size=8, shuffle=True)

input_size = len(all_words)
hidden_size = 8
output_size = len(tags)

model = NerualNet(input_size, hidden_size, output_size)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

epochs = 300

for epoch in range(epochs):
    for (words, labels) in loader:
        outputs = model(words)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    if (epoch+1) % 50 == 0:
        print(f"Epoch: {epoch+1}/{epochs}, Loss={loss.item():.4f}")

print("Training Complete.")

if os.name == "nt":
    DATA_FILE = "00_simple\\data.pth"
else:
    DATA_FILE = "./data.pth"

torch.save({
    "model_state": model.state_dict(),
    "input_size": input_size,
    "hidden_size": hidden_size,
    "output_size": output_size,
    "all_words": all_words,
    "tags": tags
}, DATA_FILE)