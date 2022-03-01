# Expired Tomatoes

This project is to demonstrate the applications of deep learning in text classification. For starters, we will identify if a movie review is a **positive** or **negative** review.
Here's an example review:
> Some may go for a film like this but I most assuredly did not. A college professor, David Norwell, suddenly gets a yen for .....

Using tensorflow's **Bidirectional Encoder Representations from Transformers (BERT)** we will able to analyze and predict that this review is in fact a **negative** review.

# How to use
_The basic implementation is shown in the [Test ExpiredTomatoes](https://github.com/lloyd-axe/Expired-Tomatoes/blob/master/Test%20ExpiredTomatoes.ipynb) notebook._

First, you'll have to install all the required packages
```
pip install -r requirements.txt
```

Then, the next steps are pretty straight forward. Basically, you'll have to define your model's path when creating an **ExpiredTomatoes** object.
After that, you can simply use the object's **get_sentiment** method to predict the sentiment. Please note that the **get_sentiment** method requires the input data to be a list of sentences.
```python
classifier = ExpiredTomatoes(MODEL_PATH)
raw = "Some may go for a film like this but I most assuredly did not. A college professor, David Norwell, suddenly gets a yen for ....."
sentences = raw.split('. ')
print('Good') if classifier.get_sentiment(sentences) == 1 else print('Bad')
```


