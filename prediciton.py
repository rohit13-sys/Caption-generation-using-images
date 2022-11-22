import string
import numpy as np
import PIL.Image

from os import listdir
from pickle import dump, load

from numpy import array
from numpy import argmax

from keras.applications.vgg16 import VGG16, preprocess_input
from keras.preprocessing.image import load_img, img_to_array
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.layers.merge import add
from keras.models import Model, load_model
from keras.layers import Input, Dense, LSTM, Embedding, Dropout
from keras.callbacks import ModelCheckpoint

from nltk.translate.bleu_score import corpus_bleu



# Function for loading a file into memory and returning text from it
def load_file(filename):
    file = open(filename, 'r')
    text = file.read()
    file.close()
    return text

# Function for loading a pre-defined list of photo identifiers
def load_photo_identifiers(filename):

    # Loading the file containing the list of photo identifier
    file = load_file(filename)

    # Creating a list for storing the identifiers
    photos = list()

    # Traversing the file one line at a time
    for line in file.split('\n'):
        if len(line) < 1:
            continue

        # Image name contains the extension as well but we need just the name
        identifier = line.split('.')[0]

        # Adding it to the list of photos
        photos.append(identifier)

    # Returning the set of photos created
    return set(photos)


# loading the cleaned descriptions that we created earlier
# we will only be loading the descriptions of the images that we will use for training
# hence we need to pass the set of train photos that the above function will be returning

def load_clean_descriptions(filename, photos):

    #loading the cleaned description file
    file = open(filename,'r')

    #creating a dictionary of descripitions for storing the photo to description mapping of train images
    descriptions = dict()

    #traversing the file line by line
    for line in file.readlines():
        # splitting the line at white spaces
        words = line.split()

        # the first word will be the image name and the rest will be the description of that particular image
        image_id, image_description = words[0], words[1:]

        # we want to load only those description which corresponds to the set of photos we provided as argument
        if image_id in photos:
            #creating list of description if needed
            if image_id not in descriptions:
                descriptions[image_id] = list()

            #the model we will develop will generate a caption given a photo,
            #and the caption will be generated one word at a time.
            #The sequence of previously generated words will be provided as input.
            #Therefore, we will need a ‘first word’ to kick-off the generation process
            #and a ‘last word‘ to signal the end of the caption.
            #we will use 'startseq' and 'endseq' for this purpose
            #also we have to convert image description back to string

            desc = 'startseq ' + ' '.join(image_description) + ' endseq'
            descriptions[image_id].append(desc)

    return descriptions

# function to load the photo features created using the VGG16 model
def load_photo_features(filename, photos):

    #this will load the entire features
    all_features = load(open(filename, 'rb'))

    #we are interested in loading the features of the required photos only
    features = {k: all_features[k] for k in photos}

    return features





# convert a dictionary of clean descriptions to a list of descriptions
def to_lines(descriptions):
    all_desc = list()
    for key in descriptions.keys():
        [all_desc.append(d) for d in descriptions[key]]
    return all_desc

# Given the descriptions, fit a tokenizer

# TOKENIZER CLASS:
# This class allows to vectorize a text corpus,
# by turning each text into either a sequence of integers
# (each integer being the index of a token in a dictionary)
# or, into a vector where the coefficient for each token could be binary, based on word count, based on tf-idf...

def create_tokenizer(descriptions):
    lines = to_lines(descriptions)
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(lines)
    return tokenizer



filename = 'Flickr_8k.trainImages.txt'
train = load_photo_identifiers(filename)
print('Dataset: ', len(train))
train_descriptions = load_clean_descriptions('descriptions.txt', train)
print('Descriptions: train=', len(train_descriptions))
tokenizer = create_tokenizer(train_descriptions)
dump(tokenizer, open('tokenizer.pkl', 'wb'))




model = VGG16()
# Removing the last layer from the loaded model as we require only the features not the classification
model.layers.pop()
model = Model(inputs=model.inputs, outputs=model.layers[-2].output)
#model.summary()


def extract_features(filename):
    image = load_img(filename, target_size=(224, 224))
    image = img_to_array(image)
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    image = preprocess_input(image)
    feature = model.predict(image, verbose=0)
    return feature

def word_for_id(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None

def generate_desc(model, tokenizer, photo, max_length):
    in_text = 'startseq'
    for i in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=max_length)
        yhat = model.predict([photo,sequence], verbose=0)
        yhat = argmax(yhat)
        word = word_for_id(yhat, tokenizer)
        if word is None:
            break
        in_text += ' ' + word
        if word == 'endseq':
            break
    return in_text
