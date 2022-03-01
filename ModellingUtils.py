
import tensorflow as tf
import tensorflow_hub as hub
import matplotlib.pyplot as plt
'''
Module specificially designed to collect data for text classifications tasks.
Dataset structure is based on the Imdb Dataset (aclImdb) for movie reviews.
'''
class ModellingUtils:
    @staticmethod
    def text_dataset_from_directory(
        path, 
        batch_size = 5, 
        validation_split = 0.2,
        subset = 'training',
        seed = 42, 
        cache = True, 
        validation = True):
        ds = tf.keras.utils.text_dataset_from_directory(
            path,
            batch_size=batch_size,
            validation_split=validation_split,
            subset = subset,
            seed = seed)

        val_ds = None
        if validation:
            val_ds = tf.keras.utils.text_dataset_from_directory(
                path,
                batch_size=batch_size,
                validation_split=validation_split,
                subset='validation',
                seed = seed)
        class_names = ds.class_names
        if cache:
            AUTOTUNE = tf.data.AUTOTUNE
            ds = ds.cache().prefetch(buffer_size=AUTOTUNE)
            if val_ds:
                val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
        return ds, class_names, val_ds

    @staticmethod
    def build_classifier_model(
        preprocess_url,
        model_url
        ):
        input_layer = tf.keras.layers.Input(shape=(), dtype=tf.string, name='text')
        preprocessing_layer = hub.KerasLayer(preprocess_url, name='preprocessing')
        encoder_inputs = preprocessing_layer(input_layer)
        encoder = hub.KerasLayer(model_url, trainable=True, name='BERT_encoder')
        outputs = encoder(encoder_inputs)
        net = outputs['pooled_output']
        net = tf.keras.layers.Dropout(0.1)(net)
        net = tf.keras.layers.Dense(1, activation=None, name='classifier')(net)
        model = tf.keras.Model(input_layer, net)
        model.compile(optimizer='Adam', loss = 'binary_crossentropy', metrics = ['binary_accuracy'])
        return model

    @staticmethod
    def train_model(
        model, 
        data, 
        epochs, 
        validation_data = None,
        plot = True):
        history = model.fit(x = data,
                            validation_data = validation_data,
                            epochs = epochs)
        if plot:
            history_dict = history.history

            acc = history_dict['binary_accuracy']
            val_acc = history_dict['val_binary_accuracy']
            loss = history_dict['loss']
            val_loss = history_dict['val_loss']

            epochs = range(1, len(acc) + 1)
            fig = plt.figure(figsize=(10, 6))
            fig.tight_layout()

            plt.subplot(2, 1, 1)
            # r is for "solid red line"
            plt.plot(epochs, loss, 'r', label='Training loss')
            # b is for "solid blue line"
            plt.plot(epochs, val_loss, 'b', label='Validation loss')
            plt.title('Training and validation loss')
            # plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.legend()

            plt.subplot(2, 1, 2)
            plt.plot(epochs, acc, 'r', label='Training acc')
            plt.plot(epochs, val_acc, 'b', label='Validation acc')
            plt.title('Training and validation accuracy')
            plt.xlabel('Epochs')
            plt.ylabel('Accuracy')
            plt.legend(loc='lower right')
        return model
