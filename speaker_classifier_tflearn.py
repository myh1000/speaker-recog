#!/usr/bin/env python
#!/usr/local/bin/python
# initial code from https://raw.githubusercontent.com/pannous/tensorflow-speech-recognition/master/speaker_classifier_tflearn.py
import os
import tensorflow as tf
import tflearn
import speech_data as data

import tensorflow as tf
print("You are using tensorflow version "+ tf.__version__) #+" tflearn version "+ tflearn.version)

# path='data/spoken_numbers_pcm/'
path='data/people/'
number_classes=0
speakers=None
model=None
speakers = data.get_speakers(path)
number_classes=len(speakers)
print("speakers",number_classes,speakers)

# Classification
tf.reset_default_graph()
tflearn.init_graph(num_cores=8, gpu_memory_fraction=0.5)

net = tflearn.input_data(shape=[None, 8192]) #Two wave chunks
net = tflearn.fully_connected(net, 64)
net = tflearn.dropout(net, 0.5)
net = tflearn.fully_connected(net, number_classes, activation='softmax')
net = tflearn.regression(net, optimizer='adam', loss='categorical_crossentropy')
model = tflearn.DNN(net)
model.load('srecog.tflearn')

def train():
	"""Trains the neural network"""
	global speakers, number_classes
	speakers = data.get_speakers(path)
	number_classes=len(speakers)
	print("speakers",speakers)

	# Classification
	tf.reset_default_graph()
	tflearn.init_graph(num_cores=8, gpu_memory_fraction=0.5)

	net = tflearn.input_data(shape=[None, 8192]) #Two wave chunks
	net = tflearn.fully_connected(net, 64)
	net = tflearn.dropout(net, 0.5)
	net = tflearn.fully_connected(net, number_classes, activation='softmax')
	net = tflearn.regression(net, optimizer='adam', loss='categorical_crossentropy')
	global model
	model = tflearn.DNN(net)
	batch=data.wave_batch_generator(batch_size=1000, source=data.Source.DIGIT_WAVES, target=data.Target.speaker,path=path)
	X,Y=next(batch)
	model.fit(X, Y, n_epoch=100, show_metric=True, snapshot_step=100)
	model.save('srecog.tflearn')

def test(fname):
	"""Predicts the person talking in the wav file

	Parameters
	----------
	fname : file name, wav format
	"""

	result=data.load_wav_file(path+fname)
	result=model.predict([result])
	print(result)
	result=data.one_hot_to_item(result,speakers)
	print("predicted speaker for %s : result = %s "%(fname,result))
	return result

if __name__ == '__main__':
	command = raw_input("What to do\n")
	if command == 'train':
		train()
	elif command == 'test':
		demo_file = "personname.wav.ig"
		test(demo_file)
