import warnings
warnings.filterwarnings('ignore')
import re
import pickle
import pandas as pd
from nltk.corpus import stopwords
from sklearn.preprocessing import MultiLabelBinarizer
from collections import Counter

TOP_50 = 50

pattern = re.compile(r'\b('+r'|'.join(stopwords.words('english'))+r')\b\s*')
def clean(text):
    text = text.lower()
    text = pattern.sub("", text)
    text = re.sub("[[**].*[**]]", "", text)
    text = re.sub("\n", " ", text)
    text = re.sub(r"[^A-Za-z^']", " ", text)
    return text

# Diagnosis codes table
diagnosis_codes = pd.read_csv('/data/diagnosis_codes.csv', sep=',')
diagnosis_codes.sort_values("SUBJECT_ID", inplace=True)

# Discharge notes table
discharge_notes_all = pd.read_csv('/data/discharge_notes.csv', sep=',')

# Aggregate and transform
# Merge and join tables
raw = pd.merge(discharge_notes_all, diagnosis_codes, left_on="HADM_ID", right_on="HADM_ID")
raw.sort_values("HADM_ID", inplace=True)

# The data
txt_cols = ['HADM_ID', 'TEXT']
txt = raw[txt_cols]
txt.sort_values("HADM_ID", inplace=True)
txt.drop_duplicates(subset="TEXT", keep='first', inplace=True)
txt.iloc[:, 1] = list(map(clean, txt.iloc[:, 1]))
txt.iloc[:, 1] = list(map(lambda text: ' '.join(text.split()), txt.iloc[:, 1]))

lbl_cols = ['HADM_ID', 'ICD9_CODE']
lbl = raw[lbl_cols]
counter = Counter(lbl.ICD9_CODE.values)
top50_codes = set(map(lambda x: x[0], counter.most_common(TOP_50)))
lbl = lbl[lbl.apply(lambda row: row.ICD9_CODE in top50_codes, axis=1)]
unique_labels = lbl['ICD9_CODE'].unique()
lbl = lbl.groupby(['HADM_ID'])['ICD9_CODE'].apply(set).reset_index()

dataset = pd.merge(txt, lbl, left_on="HADM_ID", right_on="HADM_ID")
y = dataset.ICD9_CODE
mlb_50 = MultiLabelBinarizer(classes=unique_labels)
y50 = mlb_50.fit_transform(y)
print (dataset.shape)

print("Pickling raw input data...")
x = dataset.TEXT
with open('/data/x_raw.pickle', 'wb') as f:
        pickle.dump(x, f)

print("Pickling raw output data...")
with open('/data/y_raw.pickle', 'wb') as f:
        pickle.dump(y50, f)


