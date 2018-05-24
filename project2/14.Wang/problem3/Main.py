import numpy as np
import os
import matplotlib.pyplot as plt
import pickle
import pandas as pd

from keras.preprocessing import image, sequence
from keras.applications import VGG16
from keras.layers import Dense, Convolution2D, Dropout, LSTM, TimeDistributed, Embedding, Bidirectional, Activation, RepeatVector, Merge
from keras.models import Sequential, Model
from keras.optimizers import Nadam

images_dir = os.listdir("./Flickr8k_Dataset/Flicker8k_Dataset/")

images_path = './Flickr8k_Dataset/Flicker8k_Dataset/'
captions_path = './Flickr8k_text/Flickr8k.token.txt'
train_path = './Flickr8k_text/Flickr_8k.trainImages.txt'
val_path = './Flickr8k_text/Flickr_8k.devImages.txt'

captions = open(captions_path, 'r').read().split("\n")
x_train = open(train_path, 'r').read().split("\n")
x_test = open(val_path, 'r').read().split("\n")

tokens = {}

for ix in range(len(captions)):
    temp = captions[ix].split("#")
    if temp[0] in tokens:
        tokens[temp[0]].append(temp[1][2:])
    else:
        tokens[temp[0]] = [temp[1][2:]]

temp = captions[100].split("#")
from IPython.display import Image, display
z = Image(filename=images_path+temp[0])
display(z)

for ix in range(len(tokens[temp[0]])):
    print tokens[temp[0]][ix]

print "Number of Training Images {}".format(len(x_train))

def preprocess_input(img):
    img = img[:, :, :, ::-1] #RGB to BGR
    img[:, :, :, 0] -= 103.939 
    img[:, :, :, 1] -= 116.779
    img[:, :, :, 2] -= 123.68
    return img

def preprocessing(img_path):
    im = image.load_img(img_path, target_size=(224,224,3))
    im = image.img_to_array(im)
    im = np.expand_dims(im, axis=0)
    im = preprocess_input(im)
    return im

vgg = VGG16(weights='imagenet', include_top=True, input_shape=(224,224,3))
vgg = Model(inputs=vgg.input, outputs=vgg.layers[-2].output)

def get_encoding(model, img):
    image = preprocessing(images_path+img)
    pred = model.predict(image)
    pred = np.reshape(pred, pred.shape[1])
    return pred

# # Building Vocabulary #
pd_dataset = pd.read_csv("./../Flickr8k_text/flickr_8k_train_dataset.txt", delimiter='\t')
ds = pd_dataset.values
print ds.shape

sentences = []
for ix in range(ds.shape[0]):
    sentences.append(ds[ix, 1])
    
print len(sentences)

words = [i.split() for i in sentences]

unique = []
for i in words:
    unique.extend(i)

print len(unique)

unique = list(set(unique))
print len(unique)

vocab_size = len(unique)

word_2_indices = {val:index for index, val in enumerate(unique)}
indices_2_word = {index:val for index, val in enumerate(unique)}
print word_2_indices['<start>']
print indices_2_word[4011]

max_len = 0

for i in sentences:
    i = i.split()
    if len(i) > max_len:
        max_len = len(i)

print max_len

#Model
captions = np.load("./captions.npy")
next_words = np.load("./next_words.npy")

print captions.shape
print next_words.shape

images = np.load("./images.npy")

print images.shape

images = np.load("./image_names.npy")
        
print len(image_names)

embedding_size = 128

#image model
image_model = Sequential()
image_model.add(Dense(embedding_size, input_shape=(4096,), activation='relu'))
image_model.add(RepeatVector(max_len))
image_model.summary()

#language model
language_model = Sequential()
language_model.add(Embedding(input_dim=vocab_size, output_dim=embedding_size, input_length=max_len))
language_model.add(LSTM(256, return_sequences=True))
language_model.add(TimeDistributed(Dense(embedding_size)))
language_model.summary()

#LSTM model
model = Sequential()
model.add(Merge([image_model, language_model], mode='concat', concat_axis=-1))
model.add(LSTM(1000, return_sequences=False))
model.add(Dense(vocab_size))
model.add(Activation('softmax'))

model.load_weights("./model_weights.h5")
model.compile(loss='categorical_crossentropy', optimizer=Nadam(), metrics=['accuracy'])
model.summary()

model.fit([images, captions], next_words, batch_size=512, epochs=50)

model.save_weights("./model_weights.h5")

#Prediction on test images
img = "3320032226_63390d74a6.jpg"

test_img = get_encoding(vgg, img)

z = Image(filename=images_path+img)
display(z)

def predict_captions(image):
    start_word = ["<start>"]
    while True:
        par_caps = [word_2_indices[i] for i in start_word]
        par_caps = sequence.pad_sequences([par_caps], maxlen=max_len, padding='post')
        preds = model.predict([np.array([image]), np.array(par_caps)])
        word_pred = indices_2_word[np.argmax(preds[0])]
        start_word.append(word_pred)
        
        if word_pred == "<end>" or len(start_word) > max_len:
            break
            
    return ' '.join(start_word[1:-1])

Argmax_Search = predict_captions(test_img)

def beam_search_predictions(image, beam_index = 3):
    start = [word_2_indices["<start>"]]
    
    start_word = [[start, 0.0]]
    
    while len(start_word[0][0]) < max_len:
        temp = []
        for s in start_word:
            par_caps = sequence.pad_sequences([s[0]], maxlen=max_len, padding='post')
            preds = model.predict([np.array([image]), np.array(par_caps)])
            
            word_preds = np.argsort(preds[0])[-beam_index:] #Top n prediction
            
            for w in word_preds: #new list so as to feed it to model again
                next_cap, prob = s[0][:], s[1]
                next_cap.append(w)
                prob += preds[0][w]
                temp.append([next_cap, prob])
                    
        start_word = temp
        start_word = sorted(start_word, reverse=False, key=lambda l: l[1])
        start_word = start_word[-beam_index:] # Top n words
    
    start_word = start_word[-1][0]
    intermediate_caption = [indices_2_word[i] for i in start_word]

    final_caption = []
    
    for i in intermediate_caption:
        if i != '<end>':
            final_caption.append(i)
        else:
            break
    
    final_caption = ' '.join(final_caption[1:])
    return final_caption


Beam_Search_index_3 = beam_search_predictions(test_img, beam_index=3)
Beam_Search_index_5 = beam_search_predictions(test_img, beam_index=5)
Beam_Search_index_7 = beam_search_predictions(test_img, beam_index=7)

print "Agrmax Prediction : ",
print Argmax_Search
print "Beam Search Prediction with Index = 3 : ",
print Beam_Search_index_3
print "Beam Search Prediction with Index = 5 : ",
print Beam_Search_index_5
print "Beam Search Prediction with Index = 7 : ",
print Beam_Search_index_7

display(z)


#generate captions for test images
import nltk
#bleu score
def bleu_score(hypotheses, references):
return nltk.translate.bleu_score.corpus_bleu(references, hypotheses)

test_img_dir = './Flickr8k_text/Flickr_8k.testImages.txt'

imgs = []
captions = {}

with open(test_img_dir, 'rb') as f_images:
    imgs = f_images.read().strip().split('\n')

#predict captions
f_pred_caption = open('predicted_captions.txt', 'wb')
for count, img_name in enumerate(imgs):
    print "Predicting for image: "+img_name
    test_img = get_encoding(vgg, img_name)
    Argmax_Search = predict_captions(test_img)
    Beam_Search_index_3 = beam_search_predictions(test_img, beam_index=3)
    Beam_Search_index_5 = beam_search_predictions(test_img, beam_index=5)
    Beam_Search_index_7 = beam_search_predictions(test_img, beam_index=7)
    print "Agrmax Prediction : ",
    print Argmax_Search
    print "Beam Search Prediction with Index = 3 : ",
    print Beam_Search_index_3
    print "Beam Search Prediction with Index = 5 : ",
    print Beam_Search_index_5
    print "Beam Search Prediction with Index = 7 : ",
    print Beam_Search_index_7
    captions[img_name] = Beam_Search_index_3
    f_pred_caption.write(img_name+"\t"+str(Beam_Search_index_3))
    f_pred_caption.flush()

f_pred_caption.close()

#build map of reference captions
f_captions = open('./Flickr8k_text/Flickr8k.token.txt', 'rb')
captions_text = f_captions.read().strip().split('\n')
image_captions_pair = {}
for row in captions_text:
    row = row.split("\t")
    row[0] = row[0][:len(row[0])-2]
    try:
        image_captions_pair[row[0]].append(row[1])
    except:
        image_captions_pair[row[0]] = [row[1]]      
f_captions.close()

#calculate bleu score to evaluate model performance
hypotheses=[]
references = []
for img_name in imgs:
    hypothesis = captions[img_name]
    reference = image_captions_pair[img_name]
    hypotheses.append(hypothesis)
    references.append(reference)

return bleu_score(hypotheses, references)