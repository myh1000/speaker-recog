#!/usr/bin/env python
#!/usr/local/bin/python
# initial code from https://raw.githubusercontent.com/pannous/tensorflow-speech-recognition/master/speech_data.py
"""Utilities for downloading and providing data from openslr.org, libriSpeech, Pannous, Gutenberg, WMT, tokenizing, vocabularies."""


import os
import re
import sys
import wave

import numpy
import numpy as np
import scipy
import skimage.io  # scikit-image

from random import shuffle
from six.moves import urllib
from six.moves import xrange  # pylint: disable=redefined-builtin

# TRAIN_INDEX='train_words_index.txt'
# TEST_INDEX='test_words_index.txt'
SOURCE_URL = 'http://pannous.net/files/' #spoken_numbers.tar'
DATA_DIR = 'data/'
pcm_path = "data/spoken_numbers_pcm/" # 8 bit
wav_path = "data/spoken_numbers_wav/" # 16 bit s16le
path = pcm_path
CHUNK = 4096
test_fraction=0.1 # 10% of data for test / verification

# http://pannous.net/files/spoken_numbers_pcm.tar
class Source:  # labels
	DIGIT_WAVES = 'spoken_numbers_pcm.tar'
	DIGIT_SPECTROS = 'spoken_numbers_spectros_64x64.tar'  # 64x64  baby data set, works astonishingly well
	NUMBER_WAVES = 'spoken_numbers_wav.tar'
	NUMBER_IMAGES = 'spoken_numbers.tar'  # width=256 height=256
	WORD_SPECTROS = 'https://dl.dropboxusercontent.com/u/23615316/spoken_words.tar'  # width,height=512# todo: sliding window!
	WORD_WAVES = 'spoken_words_wav.tar'
	TEST_INDEX = 'test_index.txt'
	TRAIN_INDEX = 'train_index.txt'

from enum import Enum
class Target(Enum):  # labels
	digits=1
	speaker=2
	words_per_minute=3
	word_phonemes=4
	word = 5  # int vector as opposed to binary hotword
	sentence=6
	sentiment=7
	first_letter=8
	hotword = 9
	# test_word=9 # use 5 even for speaker etc


num_characters = 32
# num_characters=60 #  only one case, Including numbers
# num_characters=128 #
# num_characters=256 #  including special characters
# offset=0  # 1:1 mapping ++
# offset=32 # starting with ' ' space
# offset=48 # starting with  numbers
offset = 64  # starting with characters
max_word_length = 20
terminal_symbol = 0

def speaker(filename):
	# if not "_" in file:
	#   return "Unknown"
	return filename.split("_")[1].lower()

def get_speakers(path=pcm_path):
	files = os.listdir(path)
	def nobad(name):
		return "_" in name and not "." in name.split("_")[1] and not 'DS_Store' in name
	speakers=list(set(map(speaker,filter(nobad,files))))
	# print(len(speakers)," speakers: ",speakers)
	return speakers

def load_wav_file(name):
	# f = wave.open(name, "rb")
	# print("loading %s"%name)
	chunk = []
	# data0 = f.readframes(CHUNK)
	# while data0:  # f.getnframes()
	# 	# data=numpy.fromstring(data0, dtype='float32')
	# 	# data = numpy.fromstring(data0, dtype='uint16')
	# 	data = numpy.fromstring(data0, dtype='uint8')
	# 	data = (data + 128) / 255.  # 0-1 for Better convergence
	# 	# chunks.append(data)
	# 	chunk.extend(data)
	# 	data0 = f.readframes(CHUNK)
	# # finally trim:
	data = scipy.io.wavfile.read(name)[1]
	data = (data + 128) / 255.
	chunk.extend(data)
	chunk = chunk[0:CHUNK * 2]  # should be enough for now -> cut
	chunk.extend(numpy.zeros(CHUNK * 2 - len(chunk)))  # fill with padding 0's
	# print("%s loaded"%name)
	return chunk

# only apply to a subset of all images at one time
def wave_batch_generator(batch_size=10,source=Source.DIGIT_WAVES,target=Target.digits,path='data/spoken_numbers_pcm/'): #speaker
	if target == Target.speaker: speakers=get_speakers(path)
	batch_waves = []
	labels = []
	import glob
	files = glob.glob(path+'*.wav')
	# print(files)
	while True:
		shuffle(files)
		# print("loaded batch of %d files" % len(files))
		for wav in files:
			if not wav.endswith(".wav"):continue
			if target==Target.speaker: labels.append(one_hot_from_item(speaker(wav), speakers))
			else: raise Exception("todo : Target.word label!")
			chunk = load_wav_file(wav)
			batch_waves.append(chunk)
			# batch_waves.append(chunks[input_width])
			if len(batch_waves) >= batch_size:
				yield batch_waves, labels
				batch_waves = []  # Reset for next batch
				labels = []

def one_hot_to_item(hot, items):
	i=np.argmax(hot)
	item=items[i]
	return item

def one_hot_from_item(item, items):
	# items=set(items) # assure uniqueness
	x=[0]*len(items)# numpy.zeros(len(items))
	i=items.index(item)
	x[i]=1
	return x
