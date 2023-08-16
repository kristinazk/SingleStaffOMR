import tensorflow as tf


def leaky_relu(features, alpha=0.2, name=None):
    with tf.name_scope(name, "LeakyRelu", [features, alpha]):
        return tf.maximum(alpha * features, features)


def default_model_params(img_height, vocabulary_size):
    params = {
        'img_height': img_height,
        'img_width': None,
        'batch_size': 16,
        'img_channels': 1,
        'conv_blocks': 4,
        'conv_filter_n': [32, 64, 128, 256],
        'conv_filter_size': [[3,3], [3,3], [3,3], [3,3]],
        'conv_pooling_size': [[2,2], [2,2], [2,2], [2,2]],
        'rnn_units': 512,
        'rnn_layers': 2,
        'vocabulary_size': vocabulary_size
    }
    return params


def ctc_crnn(params):
    # TODO: Assert parameters

    inputs = tf.keras.layers.Input(shape=(None, params['img_height'], params['img_width'], params['img_channels']),
                                   dtype=tf.float32, name='model_input')

    width_reduction = 1
    height_reduction = 1

    # Convolutional blocks
    x = inputs
    for i in range(params['conv_blocks']):
        x = tf.keras.layers.Conv2D(filters=params['conv_filter_n'][i],
                                   kernel_size=params['conv_filter_size'][i],
                                   padding="same", activation=None)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = leaky_relu(x)
        x = tf.keras.layers.MaxPooling2D(pool_size=params['conv_pooling_size'][i],
                                         strides=params['conv_pooling_size'][i])(x)
        width_reduction *= params['conv_pooling_size'][i][1]
        height_reduction *= params['conv_pooling_size'][i][0]

    # Prepare output of conv block for recurrent blocks
    features = tf.transpose(x, perm=[2, 0, 3, 1])
    feature_dim = params['conv_filter_n'][-1] * (params['img_height'] / height_reduction)
    feature_width = inputs.shape[2] / width_reduction
    features = tf.reshape(features, (tf.cast(feature_width, 'int32'), -1, tf.cast(feature_dim, 'int32')))

    # Recurrent block
    rnn_keep_prob = tf.keras.layers.Input(dtype=tf.float32, name="keep_prob")
    rnn_hidden_units = params['rnn_units']
    rnn_hidden_layers = params['rnn_layers']

    rnn_outputs, _ = tf.keras.layers.Bidirectional(tf.keras.layers.RNN(
        [tf.keras.layers.Dropout(rnn_keep_prob)(tf.keras.layers.LSTMCell(rnn_hidden_units)) for _ in range(rnn_hidden_layers)]
    ), merge_mode='concat')(features)

    logits = tf.keras.layers.Dense(params['vocabulary_size'] + 1, activation=None)(rnn_outputs)

    # CTC Loss computation
    seq_len = tf.keras.layers.Input(tf.int32, [None], name='seq_lengths')
    targets = tf.SparseTensor(dtype=tf.int32, name='target')
    ctc_loss = tf.nn.ctc_loss(labels=targets, logits=logits, label_length=seq_len)
    loss = tf.reduce_mean(ctc_loss)

    # CTC decoding
    decoded, log_prob = tf.nn.ctc_greedy_decoder(logits, seq_len)
    # decoded, log_prob = tf.nn.ctc_beam_search_decoder(logits, seq_len, beam_width=50, top_paths=1, merge_repeated=True)

    return inputs, seq_len, targets, decoded, loss, rnn_keep_prob


