import tensorflow as tf

class ExpiredTomatoes:
    def __init__(self, model_path):
        self.model = tf.saved_model.load(model_path)

    def get_sentiment(self, sentences):
        results = tf.sigmoid(self.model(tf.constant(sentences)))
        results = [res[0].numpy() for res in results]
        ave_res = sum(results) / len(results)
        return 1 if ave_res >= 0.5 else 0


