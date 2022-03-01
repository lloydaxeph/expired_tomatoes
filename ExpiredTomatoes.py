import tensorflow as tf
'''
Module for properly using the trained model for classifying the sentement
of a statement.

Main concept is to classify a sentence. Then, if the average score of the
sentences is greater than the {thresold}, the statement is positive.
'''
class ExpiredTomatoes:
    def __init__(self, model_path):
        self.model = tf.saved_model.load(model_path)

    def get_sentiment(self, sentences, thresold = 0.60):
        results = tf.sigmoid(self.model(tf.constant(sentences)))
        results = [res[0].numpy() for res in results]
        ave_res = sum(results) / len(results)
        return 1 if ave_res >= thresold else 0


