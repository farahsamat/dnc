## Application of DNC in medical diagnosis codes assignment based on clinical texts

WHAT

This project/research was done for my major thesis in my master's degree course.

<br></br>
AIM

The work involved deep learning particularly sequential modelling studies and the application of [DNC](https://www.nature.com/articles/nature20101.epdf?author_access_token=ImTXBI8aWbYxYQ51Plys8NRgN0jAjWel9jnR3ZoTv0MggmpDmwljGswxVdeocYSurJ3hxupzWuRNeGvvXnoO8o4jTJcnAyhGuZzXJ1GEaD-Z7E6X_a9R-xqJ9TfJWBqz) in healthcare domain.

<br></br>
DATA

The data used in this project is [MIMIC III](https://www.nature.com/articles/sdata201635) clinical data.

<br></br>
IMPLEMENTATION

Each clinical text [input](https://github.com/farah-samat/dnc/blob/master/discharge_notes.csv) is vectorized using [Spacy](https://spacy.io) word embedding model. Each input has possibly more than one [label](https://github.com/farah-samat/dnc/blob/master/diagnosis_codes.csv) as usually patients are diagnosed with multiple illness categories. The vectorized inputs and their associated encoded labels are then used for training.
[Samir's](https://github.com/Mostafa-Samir/DNC-tensorflow) work on DNC is adopted and used as reference for model training. We use 1-layer 64 hidden nodes LSTM with Sigmoid activation function as our DNC base controller, RMS Prop optimizer and Sigmoid cross entropy as the cost function

<br></br>
MODEL PERFORMANCE

[Hamming loss](http://scikit-learn.org/stable/modules/generated/sklearn.metrics.hamming_loss.html) is used as the model performance metric.

<br></br>
DISCLAIMER

We provide dummy data as the real MIMIC III cannot be disclosed publicly as it [contains sensitive information though de-identified](https://mimic.physionet.org/gettingstarted/access/).
Due to time constraint, the model training does not really represent sequential data modelling as word embedding is used to represent each text input. However we compare the performance of DNC against LSTM, Random Forest and SVM.
