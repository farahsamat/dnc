import warnings
warnings.filterwarnings("ignore")
import pickle
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn import metrics

n_classes = 50

# Load data
print ("Loading data...")
with open('/data/embedding.pickle', 'rb') as f:
    input = pickle.load(f)
print (len(input))
with open('/data/y_raw.pickle', 'rb') as f:
    output = pickle.load(f)
print (len(output))
print("Data loaded\n")

# Split data
train_x, test_x = train_test_split(input, test_size=0.1, random_state=42)
train_y, test_y = train_test_split(output, test_size=0.1, random_state=42)

clf = Pipeline([('classif', RandomForestClassifier())])

clf.fit(train_x, train_y)

predicted = clf.predict(test_x)
print('Correct predictions: {:4.2f}'.format(np.mean(predicted == test_y)))

print("Accuracy:", metrics.accuracy_score(test_y, predicted))
print("Precision", metrics.precision_score(test_y, predicted, average='weighted'))
print("Recall:", metrics.recall_score(test_y, predicted, average='weighted'))
print("F1:", metrics.f1_score(test_y, predicted, average='weighted'))

# Writing the hamming loss formula
incorrect = np.not_equal(predicted, test_y)
misclass = np.count_nonzero(incorrect)
hamm_loss = (np.sum(misclass/n_classes))/len(test_x)

print ("Hamming loss:,", hamm_loss)
print("Verify hamming loss value against sklearn library:", metrics.hamming_loss(test_y, predicted))



