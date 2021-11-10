import random
import json
import torch
from model import NeuralNet
from nltk_utils import bag_of_words, tokenize
from intents import opinion
from intents import this_or_that
from intents import calculator
from intents import google_search
from intents import youtube_search
from intents import news
from intents import ip_location
from intents import find_location
from intents import download_ytvid
from intents import wiki_search
from intents import googleimage_search
from intents import alarm
from intents import temperature
from intents import find_file
from intents import system_info
import utils

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open('intents.json', 'r') as json_data:
    intents = json.load(json_data)

FILE = "data.pth"
data = torch.load(FILE)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data['all_words']
tags = data['tags']
model_state = data["model_state"]

model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

bot_name = "J.A.R.V.I.S"

system_info.main()

print("\nDo you want to use voice chat Sir ? (y/n)")
start_text = input("You: ")
if start_text == 'y':
    use_voicechat = True
if start_text == 'n':
    use_voicechat = False
print('Ok Sir')

print("\nProject J.A.R.V.I.S is ready for usage")

while True:

    try:
        if use_voicechat == False:
            sentence = input("\nYou: ")
        if use_voicechat == True:
            print(" ")
            print('Listening....')
            sentence = utils.get_audio()
        command = sentence
    except:
        print(f'ERROR: a ERROR accurred Sir.')

    sentence = tokenize(sentence)
    X = bag_of_words(sentence, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)

    output = model(X)
    _, predicted = torch.max(output, dim=1)

    tag = tags[predicted.item()]

    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]

    if prob.item() > 0.84:
        print(prob.item())
        for intent in intents['intents']:
            if tag == intent["tag"]:
                print(tag)
                response = random.choice(intent['responses'])
                if tag == 'greeting':
                    print(f"{bot_name}: {response}")
                    utils.speak(response)
                elif tag == 'opinion':
                    opinion.response(command)
                elif tag == 'thanks':
                    print(f"{bot_name}: {response}")
                    utils.speak(response)
                elif tag == 'this_or_that':
                    this_or_that.response(command)
                elif tag == 'calc':
                    calculator.response()
                elif tag == 'google_search':
                    google_search.search(command)
                elif tag == 'youtube_search':
                    youtube_search.search()
                elif tag == 'news':
                    news.news()
                elif tag == 'ip_location':
                    ip_location.location()
                elif tag == 'find_place':
                    find_location.find_place()
                elif tag == 'download_vid':
                    download_ytvid.download()
                elif tag == 'wiki_search':
                    wiki_search.search()
                elif tag == 'googleimage_search':
                    googleimage_search.main()
                elif tag == 'alarm':
                    alarm.alarm()
                elif tag == 'temp':
                    temperature.weather()
                elif tag == 'find_file':
                    find_file.find_files()
                elif tag == 'goodbye':
                    print(f"{bot_name}: {response}")
                    utils.speak(response)
                    exit()
    else:
        print(f"{bot_name}: I do not understand you Sir")
        utils.speak("I do not understand you Sir")
