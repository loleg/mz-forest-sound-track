import glob
import os
import librosa
import numpy as np
import tensorflow as tf
import sounddevice

from sklearn.preprocessing import StandardScaler

duration = 1.0  # seconds
sample_rate = 48000

parent_dir = "data"
model_path = "model/snapshot"
sub_dirs = os.listdir(parent_dir)

for n, i in enumerate(sub_dirs):
    print(n, i)

def extract_features():
    features = np.empty((0,n_dim))
    X = sounddevice.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1)
    sounddevice.wait()
    X = np.squeeze(X)
    stft = np.abs(librosa.stft(X))
    mfccs = np.array(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=8).T)
    chroma = np.array(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T)
    mel = np.array(librosa.feature.melspectrogram(X, sr=sample_rate).T)
    contrast = np.array(librosa.feature.spectral_contrast(S=stft, sr=sample_rate).T)
    tonnetz = np.array(librosa.feature.tonnetz(y=librosa.effects.harmonic(X), sr=sample_rate).T)
    ext_features = np.hstack([mfccs,chroma,mel,contrast,tonnetz])
    features = np.vstack([features,ext_features])
    return features

fit_params = np.load('fit_params.npy')
sc = StandardScaler()
sc.fit(fit_params)

n_dim = 161
n_classes = len(sub_dirs)-1
n_hidden_units_one = 256
n_hidden_units_two = 256
sd = 1 / np.sqrt(n_dim)
learning_rate = 0.01

X = tf.placeholder(tf.float32,[None,n_dim])
Y = tf.placeholder(tf.float32,[None,n_classes])

W_1 = tf.Variable(tf.random_normal([n_dim,n_hidden_units_one], mean = 0, stddev=sd))
b_1 = tf.Variable(tf.random_normal([n_hidden_units_one], mean = 0, stddev=sd))
h_1 = tf.nn.tanh(tf.matmul(X,W_1) + b_1)

W_2 = tf.Variable(tf.random_normal([n_hidden_units_one,n_hidden_units_two], mean = 0, stddev=sd))
b_2 = tf.Variable(tf.random_normal([n_hidden_units_two], mean = 0, stddev=sd))
h_2 = tf.nn.sigmoid(tf.matmul(h_1,W_2) + b_2)

W = tf.Variable(tf.random_normal([n_hidden_units_two,n_classes], mean = 0, stddev=sd))
b = tf.Variable(tf.random_normal([n_classes], mean = 0, stddev=sd))
y_ = tf.nn.softmax(tf.matmul(h_2,W) + b)

init = tf.global_variables_initializer()
saver = tf.train.Saver()

y_true, y_pred = None, None
with tf.Session() as sess:
    saver.restore(sess, model_path)
    print("Model loaded")

    sess.run(tf.global_variables())
    while 1:
        feat = extract_features()
        feat = sc.transform(feat)
        y_pred = sess.run(tf.argmax(y_, 1), feed_dict={X: feat})

        print(y_pred)
