## PREDICTING HEALTHCARE DIAGNOSIS WITH CLINICAL FREE TEXTS USING DIFFERENTIABLE NEURAL COMPUTERS​

#### What

An application of Differentiable Neural Computers (DNC), a recent breakthrough in [deep learning](http://www.deeplearningbook.org/) (LeCun & Bengio, 2015) developed by [Google DeepMind](https://www.nature.com/articles/nature20101.epdf?author_access_token=ImTXBI8aWbYxYQ51Plys8NRgN0jAjWel9jnR3ZoTv0MggmpDmwljGswxVdeocYSurJ3hxupzWuRNeGvvXnoO8o4jTJcnAyhGuZzXJ1GEaD-Z7E6X_a9R-xqJ9TfJWBqz) (Graves et al., 2016) on free-text medical notes classification for diagnosis inferencing.

Clinical text classification particularly using [Memory Neural Networks](https://arxiv.org/pdf/1410.3916v1.pdf)(Weston, Chopra & Bordes, 2014) is still underexplored. This work is targeted to  fill this particular gap and at the same time demonstrates a novel application of DNC in healthcare domain specifically diagnosis codes assignment based on clinical texts.
<br></br>
#### Aim

The outcome of this work could mimic the job of a clinical coder ([What is Clinical Coding?](https://www.synapsemedical.com.au/news/2017/05/17/what-is-clinical-coding/), 2017, [Clinical coding workforce](https://www2.health.vic.gov.au/health-workforce/health-information-workforce/clinical-coding-workforce), n.d., [Clinical coding](http://www.heti.nsw.gov.au/Programs/Clinical-Coding-Workforce-Enhancement-Project/), n.d., [Clinical coder](https://en.wikipedia.org/wiki/Clinical_coder), 2017) that is to assign standard clinical codes to each patient per admission based on EHR data for various purposes including research, funding and health care planning.
<br></br>
#### Implementation
DATA

[MIMIC-III data*](https://www.nature.com/articles/sdata201635) (Johnson et al., 2016) contains EMR data from 46,146 patients from the year 2001 to 2012. 58,362 unique hospital admissions were recorded. For this project we are interested in the clinical texts and the corresponding diagnoses which were recorded in the [NOTEEVENTS](https://mimic.physionet.org/mimictables/noteevents/) table and [DIAGNOSIS_ICD](https://mimic.physionet.org/mimictables/diagnoses_icd/) table respectively.

[Text](https://github.com/farah-samat/dnc/blob/master/discharge_notes.csv) pre-processing is done to reduce noise and to increase the speed of model training. We do essential text pre-processing including stop words, white space, numbers and special characters removal. 

We group the [codes (labels)](https://github.com/farah-samat/dnc/blob/master/diagnosis_codes.csv)** into a set that corresponds to an admission and encode them into binary numbers. The encoded labels are the ones that will be used for model training. 

We categorize our problem as a multilabel text classification since a text can be classified into more than one diagnosis codes. 
<br></br>

MODEL

The DNC copy task implementation on TensorFlow by [Google DeepMind](https://github.com/deepmind/dnc) is used as reference. We adapt the copy task script by [Mostafa Samir] (https://github.com/Mostafa-Samir/DNC-tensorflow/tree/master/tasks/copy) with a few changes to suit the nature of our problem. 

A 1- layer LSTM network with 64 hidden nodes and sigmoid activation function is set to be the DNC controller. At time-step t, an external input, x(t) and a series of read vectors at the previous time-step t−1, are concatenated and fed to the controller as X(t).  

The outputs of the controller at the current time-step t are an output vector v(t) and an interface vector 𝜀(t). The interface vector defines the interactions between the controller and the external memory component at time-step t. 

The controller output, v(t) is then concatenated with the current time-step read vectors and forms the output y(t) of the DNC model. No changes are made to the memory module and we use it as is for our DNC training. 

![High-level DNC structure](https://github.com/farah-samat/dnc/blob/master/DNC.png)
<br></br>

TRAINING

We vectorize our pre-processed text inputs using [word embedding method](https://arxiv.org/abs/1301.3781) (Mikolov et al., 2013) and split the data into 9:1 training to test set ratio. Each input is now represented by a vector of size (128, 1).  

Using batches of size one, the model is trained with gradient descent where cost, the difference between the predicted label and the true label is minimized (LeCun, Bengio & Hinton, 2015) through back propagation (Rumelhart, Hinton & Williams, 1986). We use sigmoid cross entropy activation function since our labels are binary. The model parameters are optimized using RMS Prop Optimizer (Ruder, 2017, Dauphin et al. 2015) with learning rate = 0.001 and momentum = 0.9. 
<br></br>

MODEL PERFORMANCE

Since this is a multilabel classification problem, we use [Hamming loss](http://scikit-learn.org/stable/modules/generated/sklearn.metrics.hamming_loss.html) to evaluate the model performance. The Hamming loss is the fraction of labels that are incorrectly predicted and is expressed as follows. 
![Hamming loss](https://github.com/farah-samat/dnc/blob/master/Hamming_loss.png)
<br></br>

BASELINES

We set a few models – LSTM (Hochreiter & Schmidhuber, 1997), and classic machine learning models Random Forests (RF) and Support Vector Machine (SVM) as our baselines. For the LSTM, we use the same setup as the DNC controller – a 1-layer 64 hidden nodes LSTM with sigmoid activation function, optimized by RMS Prop Optimizer (Ruder, 2017, Dauphin et al. 2015) and sigmoid cross entropy cost function. 
<br></br>

FRAMEWORK AND LIBRARIES

We use Python 3.6 on TensorFlow (1.4.1) as our deep learning framework, Scikit-learn (0.19.1), numPy (1.14.0), pandas (0.20.3) in our model building training and evaluation process. We also use nltk (3.2.4) and spaCy (2.0.5) for our text pre-processing and vectorization.
<br></br>
#### Results

We compare the average loss between LSTM and DNC.
![Average loss for DNC and LSTM](https://github.com/farah-samat/dnc/blob/master/Average_loss.png)

We also observe that DNC has the lowest Hamming loss readings over 100 iterations***.
![Hamming loss observation](https://github.com/farah-samat/dnc/blob/master/Model_performance.png)

<br></br>
#### Conclusions

We have demonstrated an application of DNC in clinical text classification and the model outperformed all our baselines in terms of performance.
<br></br>

#### Challenges and future work

Applying natural language processing on clinical text is challenging. As mentioned earlier, we remove stop words, white space, numbers and special characters during text-processing. However,[numbers can give meaningful insights in medical context](https://www3.nd.edu/~nchawla/papers/ichi16b.pdf) (Feldman, Hazekamp & Chawla, n.d.). Negation words (e.g. 'no', 'not', etc.) are commonly used in clinical text and often give critical information on patients' health. Removing negation words will result the wrong clinical inference. A lot of mispelling was observed during our data processing and those words were tokenized individually which produces a rich vocabulary.

The model training does not really represent sequential data modelling as word embedding is used to represent each text input. As such, we would like to have the correct sequential modelling for clinical text since we did not take advantage if the sequential features of DNC.

Training DNCs also takes a relatively long time and we are exploring ways to minimize the training duration.
<br></br>

------
*We provide dummy data as the real MIMIC III cannot be disclosed publicly. Visit [physionet](https://mimic.physionet.org/gettingstarted/access/) to request for access.

**Only top 50 diagnoses were included for training as the rest of the diagnoses are scarce and would not help in learning process.

***A small dataset (4000+ instances) was used to get the results in a limited time.
