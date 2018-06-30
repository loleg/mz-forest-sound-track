import glob
import os
import librosa
import numpy as np
import tensorflow as tf
import sounddevice

from sklearn.preprocessing import StandardScaler

N_DIM = 161 # number of dimensions
duration = 4.0  # seconds
sample_rate = 48000

parent_dir = "data"
model_path = "model/snapshot"
sub_dirs = os.listdir(parent_dir)

for n, i in enumerate(sub_dirs):
    print(n, i)

def extract_features():
    features = np.empty((0,n_dim))
    # Record a sound wave
    X = sounddevice.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1)
    sounddevice.wait()
    # Create vectors as in the classification
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

n_dim = N_DIM
n_classes = len(sub_dirs)
n_hidden_units_one = 256
n_hidden_units_two = 256
n_hidden_units_three = 256
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

W_3 = tf.Variable(tf.random_normal([n_hidden_units_two,n_hidden_units_three], mean = 0, stddev=sd))
b_3 = tf.Variable(tf.random_normal([n_hidden_units_three], mean = 0, stddev=sd))
h_3 = tf.nn.sigmoid(tf.matmul(h_2,W_3) + b_3 )

W = tf.Variable(tf.random_normal([n_hidden_units_two,n_classes], mean = 0, stddev=sd))
b = tf.Variable(tf.random_normal([n_classes], mean = 0, stddev=sd))
y_ = tf.nn.softmax(tf.matmul(h_3,W) + b)

init = tf.global_variables_initializer()
saver = tf.train.Saver()

predicted_class = tf.argmax(Y, 1)

y_true, y_pred = None, None
with tf.Session() as sess:
    saver.restore(sess, model_path)
    print("Model loaded")

    sess.run(tf.global_variables())
    while 1:
        feat = extract_features()
        feat = sc.transform(feat)
        predicted_classes = tf.argmax(y_, 1)
        y_pred = sess.run(predicted_classes, feed_dict={X: feat})

        # print(feat)
        # print(y_pred)

        # predictions = {
        #     'class_ids': predicted_classes[:, tf.newaxis],
        #     'probabilities': tf.nn.softmax(y_),
        #     'logits': y_,
        # }
        # print(predictions)

         # Extract the predicted label (top-1)
        # _, top_predicted_label = tf.nn.top_k(y_, k=1, sorted=False)
        # # (batch_size, k) -> k = 1 -> (batch_size)
        # print(tf.squeeze(top_predicted_label, axis=1))

        # print(accuracy.eval(feed_dict={X: feat, Y: Y}))
        # correct_prediction = tf.equal(tf.argmax(Y, 1), predicted_classes)
        # # Calculate accuracy
        # # accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        # cast_value = tf.cast(y_pred, tf.float32);
        # print(sess.run(cast_value))

        d, total = {}, 0
        for p in y_pred:
            if p == 0: continue
            if not sub_dirs[p] in d: d[sub_dirs[p]] = 0
            d[sub_dirs[p]] = d[sub_dirs[p]] + 1
            total = total + 1

        for k in d.keys():
            d[k] = d[k] / len(y_pred)
        print(d)

        # accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name='accuracy')  
