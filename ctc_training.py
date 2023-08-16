import tensorflow as tf
from primus import CTC_PriMuS
import ctc_utils
import ctc_model
import argparse

import os

# Initialize GPU memory growth
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

tf.keras.backend.clear_session()  # Clear the Keras session

parser = argparse.ArgumentParser(description='Train model.')
parser.add_argument('-corpus', dest='corpus', type=str, required=True, help='Path to the corpus.')
parser.add_argument('-set', dest='set', type=str, required=True, help='Path to the set file.')
parser.add_argument('-save_model', dest='save_model', type=str, required=True, help='Path to save the model.')
parser.add_argument('-vocabulary', dest='voc', type=str, required=True, help='Path to the vocabulary file.')
parser.add_argument('-semantic', dest='semantic', action="store_true", default=False)
args = parser.parse_args()

# Load primus
primus = CTC_PriMuS(args.corpus, args.set, args.voc, args.semantic, val_split=0.1)

# Parameterization
img_height = 128
params = ctc_model.default_model_params(img_height, primus.vocabulary_size)
max_epochs = 64000
dropout = 0.5

# Model
inputs, seq_len, targets, decoded, loss, rnn_keep_prob = ctc_model.ctc_crnn(params)
train_opt = tf.keras.optimizers.Adam().minimize(loss)

model = tf.keras.models.Model(inputs=inputs, outputs=loss)

# Training loop
for epoch in range(max_epochs):
    batch = primus.nextBatch(params)

    with tf.GradientTape() as tape:
        loss_value = model([batch['inputs'], batch['seq_lengths'], ctc_utils.sparse_tuple_from(batch['targets']), dropout])

    gradients = tape.gradient(loss_value, model.trainable_variables)
    model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    if epoch % 1000 == 0:
        # VALIDATION
        print('Loss value at epoch ' + str(epoch) + ':' + str(loss_value.numpy()))
        print('Validating...')

        validation_batch, validation_size = primus.getValidation(params)

        val_idx = 0

        val_ed = 0
        val_len = 0
        val_count = 0

        while val_idx < validation_size:
            mini_batch_feed_dict = {
                inputs: validation_batch['inputs'][val_idx:val_idx + params['batch_size']],
                seq_len: validation_batch['seq_lengths'][val_idx:val_idx + params['batch_size']],
                rnn_keep_prob: 1.0
            }

            prediction = model([mini_batch_feed_dict['inputs'], mini_batch_feed_dict['seq_len'], 1.0])
            str_predictions = ctc_utils.sparse_tensor_to_strs(prediction)

            for i in range(len(str_predictions)):
                ed = ctc_utils.edit_distance(str_predictions[i], validation_batch['targets'][val_idx + i])
                val_ed = val_ed + ed
                val_len = val_len + len(validation_batch['targets'][val_idx + i])
                val_count = val_count + 1

            val_idx = val_idx + params['batch_size']

        print('[Epoch ' + str(epoch) + '] ' + str(1. * val_ed / val_count) + ' (' + str(
            100. * val_ed / val_len) + ' SER) from ' + str(val_count) + ' samples.')
        print('Saving the model...')
        model.save_weights(args.save_model + '-' + str(epoch))
        print('------------------------------')
