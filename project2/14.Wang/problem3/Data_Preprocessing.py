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

#remove the last blank row
captions = open(captions_path, 'r').read().split("\n")[:-1]
x_train = open(train_path, 'r').read().split("\n")[:-1]
x_test = open(val_path, 'r').read().split("\n")[:-1]

#match image name and captions
tokens = {}
for ix in range(len(captions)):
    temp = captions[ix].split("#")
    if temp[0] in tokens:
        tokens[temp[0]].append(temp[1][2:])
    else:
        tokens[temp[0]] = [temp[1][2:]]

#show an example
temp = captions[0].split("#")
from IPython.display import Image, display
z = Image(filename=images_path+temp[0])
display(z)

for ix in range(len(tokens[temp[0]])):
    print tokens[temp[0]][ix]

#6000
print "Number of Training Images {}".format(len(x_train))

#image preprocessing (reshape and normalize)
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

#image preprocessing example
x = preprocessing(images_path+temp[0])
print x.shape

vgg = VGG16(weights='imagenet', include_top=True, input_shape=(224,224,3))
vgg.summary()
#remove the last layer 
vgg = Model(inputs=vgg.input, outputs=vgg.layers[-2].output)
vgg.summary()

#use VGG to extract features from input image (image encoding), the output shape is 4096 
def get_encoding(model, img):
    image = preprocessing(images_path+img)
    pred = model.predict(image)
    pred = np.reshape(pred, pred.shape[1])
    return pred

print temp[0]
print get_encoding(vgg, temp[0]).shape

train_dataset = open('./Flickr8k_text/flickr_8k_train_dataset.txt','wb')
train_dataset.write("image_id\tcaptions\n")

val_dataset = open('Flickr8k_text/flickr_8k_val_dataset.txt','wb')
val_dataset.write("image_id\tcaptions\n")

#image encoding map{image_name, encoding_vector}
train_encoded_images = {}
c_train = 0
for img in x_train:
    train_encoded_images[img] = get_encoding(vgg, img)
    for capt in tokens[img]:
        caption = "<start> "+ capt + " <end>"
        train_dataset.write(img+"\t"+caption+"\n")
        train_dataset.flush()
        c_train += 1
train_dataset.close()

test_encoded_images = {}
c_test = 0
for img in x_test:
    test_encoded_images[img] = get_encoding(vgg, img)
    for capt in tokens[img]:
        caption = "<start> "+ capt + " <end>"
        val_dataset.write(img+"\t"+caption+"\n")
        val_dataset.flush()
        c_test += 1
val_dataset.close()

with open( "train_encoded_images.p", "wb" ) as pickle_f:
    pickle.dump(train_encoded_images, pickle_f )  
    
with open( "test_encoded_images.p", "wb" ) as pickle_f:
    pickle.dump(test_encoded_images, pickle_f )  

# Building Vocabulary
pd_dataset = pd.read_csv("./Flickr8k_text/flickr_8k_train_dataset.txt", delimiter='\t')
#30000 train capions
ds = pd_dataset.values
print ds.shape

sentences = []
for ix in range(ds.shape[0]):
    sentences.append(ds[ix, 1])    
print len(sentences)

words = [i.split() for i in sentences]
#example of sentence
print words[0]
print len(words)

unique = []
for i in words:
    unique.extend(i)
#example of words
print unique[:3]
#413377 words
print len(unique)
#8253 unique word, codebook
unique = list(set(unique))
print len(unique)

vocab_size = len(unique)

#Vectorization
word_2_indices = {val:index for index, val in enumerate(unique)}
indices_2_word = {index:val for index, val in enumerate(unique)}
#example of unique word and index
print word_2_indices['<start>']
print indices_2_word[4011]

#max length of sentences (40)
max_len = 0
for i in sentences:
    i = i.split()
    if len(i) > max_len:
        max_len = len(i)
print max_len

# Creating Padded Sequences and Next Words
padded_sequences, subsequent_words = [], []
for ix in range(ds.shape[0]):
    partial_seqs = []
    next_words = []
    text = ds[ix, 1].split()
    text = [word_2_indices[i] for i in text]
    for i in range(1, len(text)):
        partial_seqs.append(text[:i])
        next_words.append(text[i])
    padded_partial_seqs = sequence.pad_sequences(partial_seqs, max_len, padding='post')

    next_words_1hot = np.zeros([len(next_words), vocab_size], dtype=np.bool)
    
    #Vectorization
    for i,next_word in enumerate(next_words):
        next_words_1hot[i, next_word] = 1
        
    padded_sequences.append(padded_partial_seqs)
    subsequent_words.append(next_words_1hot)
    
padded_sequences = np.asarray(padded_sequences)
subsequent_words = np.asarray(subsequent_words)

#30000
print padded_sequences.shape
print subsequent_words.shape
#a padded matrix contains the order of words whose shape is sentence length * max_length(40)
print padded_sequences[0]

for ix in range(len(padded_sequences[0])):
    for iy in range(max_len):
        print indices_2_word[padded_sequences[0][ix][iy]],
    print "\n"
print len(padded_sequences[0])

with open('./train_encoded_images.p', 'rb') as f:
        encoded_images = pickle.load(f)

imgs = []

for ix in range(ds.shape[0]):
    imgs.append(encoded_images[ds[ix, 0]])

imgs = np.asarray(imgs)
print imgs.shape

#for simplicity, we just use the first 1500 captions to reduce the task complexity at the cost of performance
number_of_images = 1500 
#caption and next word encoding matrix
captions = np.zeros([0, max_len])
next_words = np.zeros([0, vocab_size])
for ix in range(number_of_images):#img_to_padded_seqs.shape[0]):
    captions = np.concatenate([captions, padded_sequences[ix]])
    next_words = np.concatenate([next_words, subsequent_words[ix]])
np.save("./captions.npy", captions)
np.save("./next_words.npy", next_words)

#19136 * 40
print captions.shape
#19136 * 8253
print next_words.shape

#image encoding matrix
images = []
for ix in range(number_of_images):
    for iy in range(padded_sequences[ix].shape[0]):
        images.append(imgs[ix])   
images = np.asarray(images)
np.save("./images.npy", images)
#19136 * 4096
print images.shape

image_names = []
for ix in range(number_of_images):
    for iy in range(padded_sequences[ix].shape[0]):
        image_names.append(ds[ix, 0])      
image_names = np.asarray(image_names)
np.save("./image_names.npy", image_names)
print len(image_names)

