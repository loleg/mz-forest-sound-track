import os
import glob
import librosa
import numpy as np
import tensorflow as tf

from sklearn.metrics import precision_recall_fscore_support
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import train_test_split
from sklearn.utils import shuffle

N_DIM = 161 # number of dimensions
TRAINING_EPOCHS = 5000
SAMPLE_RATE = 48000
SAMPLE_DURATION = 4.0

parent_dir = "data"
model_path = "model/snapshot"
if not os.path.exists("model"): os.makedirs("model")

sub_dirs = os.listdir(parent_dir)

def extract_features(file_name):
    X, sample_rate = librosa.load(file_name, sr=SAMPLE_RATE, duration=SAMPLE_DURATION)
    duration = librosa.core.get_duration(X)
    if duration < SAMPLE_DURATION:
        raise Exception("Too short - %d - skipping %s" %(duration, file_name))
    stft = np.abs(librosa.stft(X))
    mfccs = np.array(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=8).T)
    chroma = np.array(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T)
    mel = np.array(librosa.feature.melspectrogram(X, sr=sample_rate).T)
    contrast = np.array(librosa.feature.spectral_contrast(S=stft, sr=sample_rate).T)
    tonnetz = np.array(librosa.feature.tonnetz(y=librosa.effects.harmonic(X), sr=sample_rate).T)
    return mfccs,chroma,mel,contrast,tonnetz

def get_label(fn):
    return sub_dirs.index(fn)
    # return fn

def parse_audio_files(parent_dir,sub_dirs,file_ext='*.wav'):
    ignored = 0
    features, labels, name = np.empty((0, N_DIM)), np.empty(0), np.empty(0)
    for label, sub_dir in enumerate(sub_dirs):
        print("Processing folder..", sub_dir)
        for fn in glob.glob(os.path.join(parent_dir, sub_dir, file_ext)):
            try:
                mfccs, chroma, mel, contrast, tonnetz = extract_features(fn)
                ext_features = np.hstack([mfccs, chroma, mel, contrast, tonnetz])
                features = np.vstack([features,ext_features])
                fnlabel = get_label(sub_dir)
                l = [fnlabel] * (mfccs.shape[0])
                labels = np.append(labels, l)
            except (KeyboardInterrupt, SystemExit):
                raise
            except Exception as e:
                print(fn, e)
                ignored += 1
    print("Ignored files: ", ignored)
    return np.array(features), np.array(labels, dtype = np.int)

def one_hot_encode(labels):
    n_labels = len(labels)
    n_unique_labels = len(np.unique(labels))
    one_hot_encode = np.zeros((n_labels,n_unique_labels))
    one_hot_encode[np.arange(n_labels), labels] = 1
    return one_hot_encode

try:
    labels = np.load('labels.npy')
    features = np.load('features.npy')
    print("Features and labels found!")
except:
    print("Extracting features...")
    features, labels = parse_audio_files(parent_dir,sub_dirs)
    with open('features.npy', 'wb') as f1:
        np.save(f1, features)
    with open('labels.npy', 'wb') as f2:
        np.save(f2, labels)

labels = one_hot_encode(labels)

print("Splitting and fitting!")

train_x, test_x, train_y, test_y = train_test_split(features, labels, test_size=0.3, random_state=0)
sc = StandardScaler()
sc.fit(train_x)
with open("fit_params.npy", "wb") as f3:
    np.save(f3, train_x)
train_x = sc.transform(train_x)
test_x = sc.transform(test_x)

print("Training...")

#### Training Neural Network with TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

training_epochs = TRAINING_EPOCHS
n_dim = features.shape[1]
n_classes = len(sub_dirs)
n_hidden_units_one = 256
n_hidden_units_two = 256
n_hidden_units_three = 256
sd = 1 / np.sqrt(n_dim)
learning_rate = 0.01

X = tf.placeholder(tf.float32, [None, n_dim])
Y = tf.placeholder(tf.float32, [None, n_classes])

W_1 = tf.Variable(tf.random_normal([n_dim, n_hidden_units_one], mean=0, stddev=sd))
b_1 = tf.Variable(tf.random_normal([n_hidden_units_one], mean=0, stddev=sd))
h_1 = tf.nn.tanh(tf.matmul(X, W_1) + b_1)

W_2 = tf.Variable(tf.random_normal([n_hidden_units_one,n_hidden_units_two], mean = 0, stddev=sd))
b_2 = tf.Variable(tf.random_normal([n_hidden_units_two], mean = 0, stddev=sd))
h_2 = tf.nn.sigmoid(tf.matmul(h_1,W_2) + b_2 )

W_3 = tf.Variable(tf.random_normal([n_hidden_units_two,n_hidden_units_three], mean = 0, stddev=sd))
b_3 = tf.Variable(tf.random_normal([n_hidden_units_three], mean = 0, stddev=sd))
h_3 = tf.nn.sigmoid(tf.matmul(h_2,W_3) + b_3 )

W = tf.Variable(tf.random_normal([n_hidden_units_two, n_classes], mean=0, stddev=sd))
b = tf.Variable(tf.random_normal([n_classes], mean=0, stddev=sd))
y_ = tf.nn.softmax(tf.matmul(h_3, W) + b)

init = tf.global_variables_initializer()
saver = tf.train.Saver()

cost_function = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(y_), reduction_indices=[1]))
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost_function)

correct_prediction = tf.equal(tf.argmax(y_,1), tf.argmax(Y,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


batch_size = 10000
patience_cnt = 0
patience = 16
min_delta = 0.01
stopping = 0

cost_history = np.empty(shape=[1], dtype=float)
y_true, y_pred = None, None
with tf.Session() as sess:
    sess.run(init)
    for epoch in range(training_epochs):
        if stopping == 0:
            total_batch = (train_x.shape[0] // batch_size)
            train_x = shuffle(train_x, random_state=42)
            train_y = shuffle(train_y, random_state=42)
            for i in range(total_batch):
                batch_x = train_x[i*batch_size:i*batch_size+batch_size]
                batch_y = train_y[i*batch_size:i*batch_size+batch_size]
                # Run optimization op (backprop) and cost op (to get loss value)
                _, cost = sess.run([optimizer, cost_function], feed_dict={X: batch_x, Y: batch_y})
            cost_history = np.append(cost_history, cost)
            if epoch % 100 == 0:
                print("Epoch: ", epoch, " cost ", cost)
            if epoch > 0 and abs(cost_history[epoch-1] - cost_history[epoch]) > min_delta:
                patience_cnt = 0
            else:
                patience_cnt += 1
            if patience_cnt > patience:
                print("Early stopping at epoch ", epoch, ", cost ", cost)
                stopping = 1

    y_pred = sess.run(tf.argmax(y_,1),feed_dict={X: test_x})
    y_true = sess.run(tf.argmax(test_y,1))

    # Saving model https://www.tensorflow.org/api_docs/python/tf/train/Saver
    save_path = saver.save(sess, model_path)
    print("Model saved at: %s" % save_path)

p,r,f,s = precision_recall_fscore_support(y_true, y_pred)#average='micro')
print ("F-Score:", f)
print ("Precision:", p)
print ("Recall:", r)
