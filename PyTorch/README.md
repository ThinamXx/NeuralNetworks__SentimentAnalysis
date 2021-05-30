# **Natural Language Processing and Sentiment Analysis with Neural Networks**

**Sentiment Analysis**
- Sentiment Analysis is the use of natural language processing, text analysis, computational linguistics, and biometrics to systematically identify, extract, quantify and study affective states and subjective information. Sentiment analysis is widely applied to voice of the customer materials such as reviews and survey responses, online and social media and healthcare materials for applications that range from marketing to customer service to clinical medicine.

**Notebooks:**
- [**Sentiment Analysis Dataset Notebook**](https://github.com/ThinamXx/NeuralNetworks__SentimentAnalysis/blob/master/PyTorch/Sentiment%20Analysis%20Dataset.ipynb)
- [**Sentiment Analysis using RNN**](https://github.com/ThinamXx/NeuralNetworks__SentimentAnalysis/blob/master/PyTorch/Sentiment%20Analysis%20RNN.ipynb)
- [**Sentiment Analysis using CNN**](https://github.com/ThinamXx/NeuralNetworks__SentimentAnalysis/blob/master/PyTorch/Sentiment%20Analysis%20CNN.ipynb)

**Getting the Dataset**
- I have used Google Colab for this project so the process of downloading and reading the Data might be different in other platforms. I have used [**Large Movie Review Dataset**](https://ai.stanford.edu/~amaas/data/sentiment/) for this project  which was compiled for the 2011 paper Learning Word Vectors for Sentiment Analysis. The dataset is divided into training and testing and each contains 25000 movie reviews. I have presented the implementation of Reading the Dataset, Tokenization and Vocabulary and Padding to Fixed Length using PyTorch here in the snapshot.

![Image](https://github.com/ThinamXx/300Days__MachineLearningDeepLearning/blob/main/Images/Day%20176.PNG)

**Recurrent Neural Network Model**
- Each words obtains a feature vector from the embedding layer which is further encoded using bidirectional RNN to obtain sequence information. Here the Embedding instance is the embedding layer, the LSTM instance is the hidden layer for sequence encoding and the Dense instance is the output layer for generated classification result.  I have presented the implementation of Bidirectional Recurrent Neural Networks Model using PyTorch here in the snapshot. 

![IMAGE](https://github.com/ThinamXx/300Days__MachineLearningDeepLearning/blob/main/Images/Day%20177.PNG)

**Text Convolutional Neural Networks**
- Text CNN uses a one dimensional convolutional layer and max over time pooling layer. I will use two embedding layers: one with fixed weight and another that participates in training. I have presented the implementation of Text Convolutional Neural Networks using PyTorch here in the snapshot.

![Image](https://github.com/ThinamXx/300Days__MachineLearningDeepLearning/blob/main/Images/Day%20178.PNG)
