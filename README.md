# **Natural Language Processing and Sentiment Analysis with Neural Networks**

**Introduction and Objective**
- I have prepared a Model using [**Convolutional Neural Network**](https://github.com/ThinamXx/NeuralNetwork__SentimentAnalysis/blob/master/SentimentAnalysis%20with%20CNN.ipynb), [**Recurrent Neural Network**](https://github.com/ThinamXx/NeuralNetwork__SentimentAnalysis/blob/master/SentimentAnalysis%20with%20RNN.ipynb) and [**Long Short Term Memory**](https://github.com/ThinamXx/NeuralNetwork__SentimentAnalysis/blob/master/Sentiment%20Analysis%20with%20LSTM.ipynb) to classify the Sentiment of Text Data. In this Project, I have used the Data from [**Large Movie Review Dataset**](https://ai.stanford.edu/~amaas/data/sentiment/) which was compiled for the 2011 paper **Learning Word Vectors for Sentiment Analysis**.

**Notebooks:**
- [**Sentiment Analysis with Convolutional Neural Network**](https://github.com/ThinamXx/NeuralNetwork__SentimentAnalysis/blob/master/SentimentAnalysis%20with%20CNN.ipynb)

- [**Sentiment Analysis with Recurrent Neural Network**](https://github.com/ThinamXx/NeuralNetwork__SentimentAnalysis/blob/master/SentimentAnalysis%20with%20RNN.ipynb)

- [**Sentiment Analysis with Long Short Term Memory**](https://github.com/ThinamXx/NeuralNetwork__SentimentAnalysis/blob/master/Sentiment%20Analysis%20with%20LSTM.ipynb)


**Convolutional Neural Network**
- In Deep Learning, a Convolutional Neural Network is a class of Deep Neural Networks, most commonly applied to analyzing Visual Imagery. They are also known as shift invariant or space invariant Artificial Neural Networks, based on their shared-weights architecture and translation invariance characteristics.

**Libraries and Dependencies**

```javascript
import os, glob
from random import shuffle
from IPython.display import display

import numpy as np                                      
from keras.preprocessing import sequence                
from keras.models import Sequential                     
from keras.layers import Dense, Dropout, Activation     
from keras.layers import Conv1D, GlobalMaxPooling1D     

from nltk.tokenize import TreebankWordTokenizer         
from gensim.models.keyedvectors import KeyedVectors
from nlpia.loaders import get_data    
```

**Getting the Dataset**
- I have used Google Colab for this Project so the process of downloading and reading the Data might be different in other platforms. I have used [**Large Movie Review Dataset**](https://ai.stanford.edu/~amaas/data/sentiment/#:~:text=This%20is%20a%20dataset%20for,data%20for%20use%20as%20well.) for this Project  which was compiled for the 2011 paper **Learning Word Vectors for Sentiment Analysis**. Since, It is a very large Dataset, I have used just the subset of the Dataset. I will be Implementing Convolutional Neural Netwrok for this Project. This is a dataset for binary sentiment classification containing substantially more data. The Dataset has a set of 25,000 highly polar movie reviews for training and 25,000 for testing. There is additional unlabeled data for use as well. Raw text and already processed bag of words formats are provided. 

**Convolutional Neural Network**
- In Deep Learning, a Convolutional Neural Network is a class of Deep Neural Networks, most commonly applied to analyzing Visual Imagery. They are also known as shift invariant or space invariant Artificial Neural Networks, based on their shared-weights architecture and translation invariance characteristics. After the Data is ready to build the Network, I will build the Model using Convolutional Neural Network. Each stride in the Convolution will be of one token. And I will be using the ReLU activation Function.
- Snapshot of the Convolutional Neural Network Mode:

![Image](https://github.com/ThinamXx/66Days__NaturalLanguageProcessing/blob/master/Images/02.PNG)

**Model Evaluation**
- Snapshot of the Sentiment Analysis of the Model:

![Image](https://github.com/ThinamXx/66Days__NaturalLanguageProcessing/blob/master/Images/O3.PNG)

**Saving the Model**

```javascript
model_structure = model.to_json()                            
with open("cnn_model.json", "w") as json_file:
  json_file.write(model_structure)
model.save_weights("cnn_weights.h5")
```

