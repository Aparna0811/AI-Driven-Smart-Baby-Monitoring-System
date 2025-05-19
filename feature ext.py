import librosa
import librosa.display
import IPython.display as ipd
import os
import numpy as np
for file in os.listdir('D:\\audio data'):
    print(file)
ipd.Audio('D:\\audio data\\dataset2-20241128T125237Z-001\\dataset2\\cry\\1c.wav')
def feature_extraction(file_path):
    # load the audio file
    x, sample_rate = librosa.load(file_path, res_type='kaiser_fast')
    # extract features from the audio
    mfcc = np.mean(librosa.feature.mfcc(y=x, sr=sample_rate, n_mfcc=50).T, axis=0)
    
    return mfcc

features = {}
directory = 'D:\\audio data'
for audio in os.listdir(directory):
    audio_path =os.path.join(directory,audio)
    features[audio_path] = feature_extraction(audio_path)

audio_path
features[audio_path], len(features[audio_path])