import random
import torch
import json
import os
from model import NeuralNet
from utils import bag_of_words, tokenize

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if os.name == "nt":
    DATA_FILE = "00_simple\\data.pth"
    INTENTS_FILE = "00_simple\\intents.json"
else:
    DATA_FILE = "./data.pth"
    INTENTS_FILE = "./intents.json"
    
data = torch.load(DATA_FILE)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data["all_words"]
tags = data["tags"]

model = NeuralNet(input_size, hidden_size, output_size)
model.load_state_dict(data["model_state"])
model.eval()

with open(INTENTS_FILE, "r") as f:
    intents = json.load(f)

print("Chatbot running! Type 'quit' to exit")

while True:
    sentence = input("You: ")
    if sentence.lower() == "quit":
        break

    X = bag_of_words(tokenize(sentence), all_words)
    X = torch.tensor(X, dtype=torch.float32)

    output = model(X)
    _, predicted = torch.max(output, dim=0)
    tag = tags[predicted.item()]

    probs = torch.softmax(output, dim=0)
    prob = probs[predicted.item()]

    if prob.item(0 > 0.75):
        for intent in intents["intents"]:
            if tag == intent["tag"]:
                print(f"Bot: {random.choice(intent["responses"])}")
    else:
        print("Bot: I'm sorry, I don't understand.")