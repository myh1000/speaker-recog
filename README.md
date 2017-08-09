# Speaker Recognition with a 2-layer classifier

Rudimentary text-dependent speaker recognition that uses 2-layer deep neural network along with cross entropy loss to classify speakers in a text-dependent environment.

speaker-recog works by taking a live recording of one's voice and trims out the silence in the recording, then running it through the neural network and predicts the speaker.

## Training/Testing

Run ```speaker_classifier_tflearn.py``` and input ```test``` or ```train``` in the command line to test or train, respectively.

Use ```srecogskill.py``` to add and preprocess new audio recordings to file.

Change ```path``` to the directory with the audio files stored (default ```data/spoken_numbers_pcm/```)
