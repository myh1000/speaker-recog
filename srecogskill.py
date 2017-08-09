import record as record
import speaker_classifier_tflearn
"""Miscellaneous testing for speaker recog"""

record.record_to_file("michael") #create data
speaker_classifier_tflearn.train() #retrain

# record.record_to_file("temp",train=False) #create data
# person = speaker_classifier_tflearn.test("temp.wav.ig") #call whenever you want to classify
