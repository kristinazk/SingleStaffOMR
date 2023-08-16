import argparse
import tensorflow as tf
import ctc_utils
import cv2
import numpy as np

parser = argparse.ArgumentParser(description='Decode a music score image with a trained model (CTC).')
parser.add_argument('-image', dest='image', type=str, required=True, help='Path to the input image.')
parser.add_argument('-model', dest='model', type=str, required=True, help='Path to the trained model.')
parser.add_argument('-vocabulary', dest='voc_file', type=str, required=True, help='Path to the vocabulary file.')
args = parser.parse_args()

# Read the dictionary
with open(args.voc_file, 'r') as dict_file:
    dict_list = dict_file.read().splitlines()
    int2word = {i: word for i, word in enumerate(dict_list)}

# Restore weights
model = tf.keras.models.load_model(args.model, compile=False)
graph = tf.compat.v1.get_default_graph()

input_tensor = graph.get_tensor_by_name("model_input:0")
seq_len_tensor = graph.get_tensor_by_name("seq_lengths:0")
rnn_keep_prob_tensor = graph.get_tensor_by_name("keep_prob:0")
height_tensor = graph.get_tensor_by_name("input_height:0")
width_reduction_tensor = graph.get_tensor_by_name("width_reduction:0")
logits = model.get_layer("dense").output  # Assuming the logits layer is named "dense"

# Constants that are saved inside the model itself
WIDTH_REDUCTION, HEIGHT = sess.run([width_reduction_tensor, height_tensor])

decoded, _ = tf.nn.ctc_greedy_decoder(logits, seq_len_tensor)

image = cv2.imread(args.image, cv2.IMREAD_GRAYSCALE)
image = ctc_utils.resize(image, HEIGHT)
image = ctc_utils.normalize(image)
image = np.expand_dims(image, axis=0)

seq_lengths = [image.shape[2] / WIDTH_REDUCTION]

prediction = model.predict([image, seq_lengths, np.ones_like(seq_lengths)])  # Assuming rnn_keep_prob is always 1.0

str_predictions = ctc_utils.sparse_tensor_to_strs(prediction)
for w in str_predictions[0]:
    print(int2word[w], end=' ')
    print('\t')
