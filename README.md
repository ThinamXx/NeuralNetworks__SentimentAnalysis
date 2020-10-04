# **Natural Language Processing and Sentiment Analysis with Neural Networks**

**Introduction and Objective**
- I have prepared a Model using [**Convolutional Neural Network**](https://github.com/ThinamXx/NeuralNetwork__SentimentAnalysis/blob/master/SentimentAnalysis%20with%20CNN.ipynb), [**Recurrent Neural Network**](https://github.com/ThinamXx/NeuralNetwork__SentimentAnalysis/blob/master/SentimentAnalysis%20with%20RNN.ipynb) and [**Long Short Term Memory**](https://github.com/ThinamXx/NeuralNetwork__SentimentAnalysis/blob/master/Sentiment%20Analysis%20with%20LSTM.ipynb) to classify the Sentiment of Text Data. In this Project, I have used the Data from [**Large Movie Review Dataset**](https://ai.stanford.edu/~amaas/data/sentiment/) which was compiled for the 2011 paper **Learning Word Vectors for Sentiment Analysis**.

**Notebooks:**
- [**Sentiment Analysis with Convolutional Neural Network (CNN)**](https://github.com/ThinamXx/NeuralNetwork__SentimentAnalysis/blob/master/SentimentAnalysis%20with%20CNN.ipynb)

- [**Sentiment Analysis with Recurrent Neural Network (RNN)**](https://github.com/ThinamXx/NeuralNetwork__SentimentAnalysis/blob/master/SentimentAnalysis%20with%20RNN.ipynb)

- [**Sentiment Analysis with Long Short Term Memory (LSTM)**](https://github.com/ThinamXx/NeuralNetwork__SentimentAnalysis/blob/master/Sentiment%20Analysis%20with%20LSTM.ipynb)

- [**Text Generation with Long Short Term Memory (LSTM)**](https://github.com/ThinamXx/NeuralNetwork__SentimentAnalysis/blob/master/Generating%20Text%20with%20LSTM.ipynb)

**Libraries and Dependencies**
- I have listed all the necessary Libraries and Dependencies required for this Project here:

```javascript
import os, glob
from random import shuffle
from IPython.display import display

import numpy as np                                      
from keras.preprocessing import sequence                
from keras.models import Sequential                     
from keras.layers import Dense, Dropout, Activation, Flatten     
from keras.layers import Conv1D, GlobalMaxPooling1D  
from keras.layers import LSTM, SimpleRNN
from keras.layers.wrappers import Bidirectional

from nltk.tokenize import TreebankWordTokenizer         
from gensim.models.keyedvectors import KeyedVectors
from nlpia.loaders import get_data    
```

**Getting the Dataset**
- I have used Google Colab for this Project so the process of downloading and reading the Data might be different in other platforms. I have used [**Large Movie Review Dataset**](https://ai.stanford.edu/~amaas/data/sentiment/) for this Project  which was compiled for the 2011 paper **Learning Word Vectors for Sentiment Analysis**. Since, It is a very large Dataset, I have used just the subset of the Dataset. This is a dataset for binary sentiment classification containing substantially more data. The Dataset has a set of 25,000 highly polar movie reviews for training and 25,000 for testing. There is additional unlabeled data for use as well. Raw text and already processed bag of words formats are provided. 

**Convolutional Neural Network**
- In Deep Learning, a Convolutional Neural Network is a class of Deep Neural Networks, most commonly applied to analyzing Visual Imagery. They are also known as shift invariant or space invariant Artificial Neural Networks, based on their shared-weights architecture and translation invariance characteristics. After the Data is processed and made ready to build the Network, I have built the Model using Convolutional Neural Network. Each stride in the Convolution is of one token. And I have used the ReLU activation Function.

- Snapshot of Convolutional Neural Network:

![Image](https://github.com/ThinamXx/66Days__NaturalLanguageProcessing/blob/master/Images/02.PNG)

**Recurrent Neural Network**
- A Recurrent Neural Network or RNN is a class of Artificial Neural Networks where connections between nodes form a directed graph along a temporal sequence. This allows it to exhibit temporal Dynamic behavior. Derived from Feedforward Neural Networks, RNN can use their internal state or memory to process variable length sequences of inputs. This makes them applicable to tasks such as Unsegmented and Connected Handwriting Recognition or Speech Recognition.

- Snapshot of Recurrent Neural Network:

![Image](https://github.com/ThinamXx/66Days__NaturalLanguageProcessing/blob/master/Images/Day%2029.PNG)

**Long Short Term Memory**
- Long Short Term Memory or LSTM is an Artificial Recurrent Neural Network or RNN architecture used in the field of Deep Learning. Unlike standard Feedforward Neural Networks, LSTM has Feedback connections. It can not only process single data points, but also entire sequences of data such as Speech or Video.

- Snapshot of Long Short Term Memory:

![Image](https://github.com/ThinamXx/66Days__NaturalLanguageProcessing/blob/master/Images/Day%2030.PNG)

[**Text Generation with LSTM**](https://github.com/ThinamXx/NeuralNetwork__SentimentAnalysis/blob/master/Generating%20Text%20with%20LSTM.ipynb)
- Long Short Term Memory or LSTM is an Artificial Recurrent Neural Network or RNN architecture used in the field of Deep Learning. Unlike standard Feedforward Neural Networks, LSTM has Feedback connections. It can not only process single data points, but also entire sequences of data such as Speech or Video. I need a Dataset which is more consistent across samples in style and tone or a much larger Dataset. The Keras Example provides a sample of the work of Friedrich Nietzsche. But I will choose someone else with a singular style : William Shakespeare.

- Snapshot of Code for Generating Text:

![Image](https://github.com/ThinamXx/66Days__NaturalLanguageProcessing/blob/master/Images/31a.PNG)

- Snapshot of the Generated Text:

![Image](https://github.com/ThinamXx/66Days__NaturalLanguageProcessing/blob/master/Images/31b.PNG)

**Model Evaluation**
- I made a sentence with a Positive sentiment. Then, I used the Model to predict the Sentiment of the Text Data. Snapshot of the same is as:

![Image](https://github.com/ThinamXx/66Days__NaturalLanguageProcessing/blob/master/Images/O3.PNG)

- Again, I made a sentence with Negative Sentiment  and used the Model to predict the Sentiment of the Text Data. Snapshot of the same is as:

![Image](https://github.com/ThinamXx/66Days__NaturalLanguageProcessing/blob/master/Images/O4.PNG)

**Saving the Model**
- The Code Snippets helps to save the Model:

```javascript
model_structure = model.to_json()                            
with open("model.json", "w") as json_file:
  json_file.write(model_structure)
model.save_weights("model.h5")
```

