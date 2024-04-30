# Import the required module for text
# to speech conversion
from gtts import gTTS
import numpy as np
# This module is imported so that we can
# play the converted audio
import os

from datetime import datetime


# The text that you want to convert to audio
mytext = 'Welcome to geeksforgeeks!'

# Language in which you want to convert
language = 'en'

# Passing the text and language to the engine,
# here we have marked slow=False. Which tells
# the module that the converted audio should
# have a high speed

mytext = "Hi I'm Sandi"
myobj = gTTS(text=mytext, lang=language, slow=False)
myobj.save("/mnt/datastore/Harry/sandi/"+mytext+".mp3")

mytext = "sophisticated Automated Name Drawing Intelligence"
myobj = gTTS(text=mytext, lang=language, slow=False)
myobj.save("/mnt/datastore/Harry/sandi/"+mytext+".mp3")

mytext = 'hello, my name is Jane, I was created to create random selections of names'
myobj = gTTS(text=mytext, lang=language, slow=False)
myobj.save("/mnt/datastore/Harry/sandi/"+mytext+".mp3")

mytext = 'James'
myobj = gTTS(text=mytext, lang=language, slow=False)
myobj.save("/mnt/datastore/Harry/sandi/"+mytext+".mp3")

mytext = 'hello'
myobj = gTTS(text=mytext, lang=language, slow=False)
myobj.save("/mnt/datastore/Harry/sandi/"+mytext+".mp3")

mytext = 'hi'
myobj = gTTS(text=mytext, lang=language, slow=False)
myobj.save("/mnt/datastore/Harry/sandi/"+mytext+".mp3")

mytext = 'yes'
myobj = gTTS(text=mytext, lang=language, slow=False)
myobj.save("/mnt/datastore/Harry/sandi/"+mytext+".mp3")

mytext = 'no'
myobj = gTTS(text=mytext, lang=language, slow=False)
myobj.save("/mnt/datastore/Harry/sandi/"+mytext+".mp3")

mytext = 'please stop'
myobj = gTTS(text=mytext, lang=language, slow=False)
myobj.save("/mnt/datastore/Harry/sandi/"+mytext+".mp3")

mytext = 'I love you'
myobj = gTTS(text=mytext, lang=language, slow=False)
myobj.save("/mnt/datastore/Harry/sandi/"+mytext+".mp3")

mytext = 'well done'
myobj = gTTS(text=mytext, lang=language, slow=False)
myobj.save("/mnt/datastore/Harry/sandi/"+mytext+".mp3")


mytext = 'that was very good'
myobj = gTTS(text=mytext, lang=language, slow=False)
myobj.save("/mnt/datastore/Harry/sandi/"+mytext+".mp3")

np.random.seed(int(datetime.now().timestamp()))

#names_in_the_hat = ["Harry", "Matt", "Chris", "Bri", "Ian", "Gerrard", "Raphael","Jinghau", "Sara",
#                    "Sau", "Kirsty", "Zita", "Roshne", "Yifang", "Wolf", "Gulsen", "Edward"]

names_in_the_hat = ["Harry", "Matt", "Chris", "Gerrard","Jinghau", "Sara",
                    "Kirsty", "Zita", "Roshne", "Yifang", "Wolf", "Edward", "Sandra", "Katie"]

selection = np.random.randint(low=0, high=len(names_in_the_hat))
mytext = 'I am now Randomising, I am now Randomising, I am now shuffling names, I have chosen a name. '+names_in_the_hat[selection]+' will be presenting, Better luck next time'
myobj = gTTS(text=mytext, lang=language, slow=False)
myobj.save("/mnt/datastore/Harry/sandi/speaker_selection1.mp3")

selection = np.random.randint(low=0, high=len(names_in_the_hat))
mytext = 'I am now Randomising, I am now Randomising, I am now shuffling names, I have chosen a name. '+names_in_the_hat[selection]+' will be explaining, Good luck'
myobj = gTTS(text=mytext, lang=language, slow=False)
myobj.save("/mnt/datastore/Harry/sandi/explainer_selection1.mp3")

selection = np.random.randint(low=0, high=len(names_in_the_hat))
mytext = 'I am now Randomising, I am now Randomising, I am now shuffling names, I have chosen a name. '+names_in_the_hat[selection]+' will be presenting, Better luck next time'
myobj = gTTS(text=mytext, lang=language, slow=False)
myobj.save("/mnt/datastore/Harry/sandi/speaker_selection2.mp3")

selection = np.random.randint(low=0, high=len(names_in_the_hat))
mytext = 'I am now Randomising, I am now Randomising, I am now shuffling names, I have chosen a name. '+names_in_the_hat[selection]+' will be explaining, Good luck'
myobj = gTTS(text=mytext, lang=language, slow=False)
myobj.save("/mnt/datastore/Harry/sandi/explainer_selection2.mp3")


print("!")
# Playing the converted file
#os.system("mpg321 welcome.mp3")
