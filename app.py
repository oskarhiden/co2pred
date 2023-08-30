import numpy as np
import pandas as pd
import os
from os.path import dirname

from scipy.spatial.distance import cosine
from sentence_transformers import SentenceTransformer
import streamlit as st
import torch
import torch.nn as nn
from sklearn.preprocessing import OneHotEncoder



def cosine_similarity(a, b):
    return 1 - cosine(a, b)

#path = dirname(os.getcwd()) + '/aica/ClimateBERT/'

#climateBERTmodel
class climateBERT(nn.Module):
    def __init__(self, bert_emb_crop, hidden_size, nr_layers):#, llm_model):
        super(climateBERT, self).__init__()
        if nr_layers == 1:
            layers = [nn.Linear(bert_emb_crop, 1)]
        else:
            layers = []
            for layer in range(nr_layers-1):
                if layer==0:
                    layers.append(nn.Linear(bert_emb_crop, hidden_size))
                else:
                    layers.append(nn.Linear(hidden_size, hidden_size))
                layers.append(nn.ReLU())
            layers.append(nn.Linear(hidden_size, 1))

        self.linear = nn.Sequential(*layers)

    def forward(self, emb_cls):
        out = self.linear(emb_cls)
        return out

def load_model(model):
    path = '' # path + 'web_pred/'
    model.load_state_dict(torch.load(path + 'stored_model.pt', map_location=torch.device('cpu')))
    return model

######################
# Load data
path_data = '' #'dirname(os.getcwd()) + '/aica/ClimateBERT/' #+ '/aica/EPD_ETL/'
path = '' # dirname(os.getcwd()) + '/aica/ClimateBERT/'
file_name_n = '4_epds_epdnorway_kg.csv'
file_name_o = '4_epds_oekobaudat_kg.csv'
df1 = pd.read_csv(path_data + file_name_n)
df2 = pd.read_csv(path_data + file_name_o)
df = pd.concat([df1.copy(), df2.copy()], ignore_index=True)
df = df[df['Unit'].isin(['Kilogram'])].copy()
print('Number of EPDs: ', len(df))

embs = []
# uuids = df.ILCD.values
# Text
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
with torch.no_grad():
    embeddings = model.encode(df.Name.values)
print('embeddings shape:', embeddings.shape)
embs.append(embeddings)

# category encoding
cat1 = df.Cat_H1.values
cat1_unique = np.unique(cat1.astype(str))
enc1 = OneHotEncoder(handle_unknown='ignore')
cat1 = enc1.fit_transform(cat1.reshape(-1, 1)).toarray()
print('cat1 shape:', cat1.shape)
embs.append(cat1)

cat2 = df.Cat_H2.values
cat2_unique = np.unique(cat2.astype(str))
enc2 = OneHotEncoder(handle_unknown='ignore')
cat2 = enc2.fit_transform(cat2.reshape(-1, 1)).toarray()
print('cat2 shape:', cat2.shape)
embs.append(cat2)

cat3 = df.Cat_H3.values
cat3_unique = np.unique(cat3.astype(str))
enc3 = OneHotEncoder(handle_unknown='ignore')
cat3 = enc3.fit_transform(cat3.reshape(-1, 1)).toarray()
print('cat3 shape:', cat3.shape)
embs.append(cat3)

# Geography encoding
geo = df.Geo.values
geo_unique = np.unique(geo.astype(str))
enc4 = OneHotEncoder(handle_unknown='ignore')
geo = enc4.fit_transform(geo.reshape(-1, 1)).toarray()
print('geo shape:', geo.shape)
embs.append(geo)

x_vals = np.concatenate(embs, axis=1)
y_vals = df.Emission_kg.values.astype(np.float32)
_, input_size = x_vals.shape
print('x_vals shape:', x_vals.shape)
######################


hidden_size = 256
nr_layers = 4
model1 = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
model = climateBERT(input_size, hidden_size, nr_layers)
model = load_model(model)

def comb_bert(text, category1, category2, category3, geo):
    embed = model1.encode(text)
    cat1 = enc1.transform(category1.reshape(-1, 1)).toarray()
    cat2 = enc2.transform(category2.reshape(-1, 1)).toarray()
    cat3 = enc3.transform(category3.reshape(-1, 1)).toarray()
    geo = enc4.transform(geo.reshape(-1, 1)).toarray()
    emb = np.concatenate([embed, cat1, cat2, cat3, geo], axis=1)

    out = model.forward(torch.tensor(emb, dtype=torch.float32))
    return out.detach().numpy()
##########3 end temporary



st.title('Carbon footprint prediction')
# category translation
cat1_translation = pd.read_excel(path + 'cat_translate.xlsx', sheet_name='Cat1') 
tyska = cat1_translation['Tyska'].values
svenska = cat1_translation['Svenska'].values
cat1_sv_to = {}
cat1_ty_to = {}
for i in range(len(tyska)):
    cat1_ty_to[tyska[i]] = svenska[i]
    cat1_sv_to[svenska[i]] = tyska[i]

cat2_translation = pd.read_excel(path + 'cat_translate.xlsx', sheet_name='Cat2')
tyska = cat2_translation['Tyska'].values
svenska = cat2_translation['Svenska'].values
cat2_sv_to = {}
cat2_ty_to = {}
for i in range(len(tyska)):
    cat2_ty_to[tyska[i]] = svenska[i]
    cat2_sv_to[svenska[i]] = tyska[i]

cat3_translation = pd.read_excel(path + 'cat_translate.xlsx', sheet_name='Cat3')
tyska = cat3_translation['Tyska'].values
svenska = cat3_translation['Svenska'].values
cat3_sv_to = {}
cat3_ty_to = {}
for i in range(len(tyska)):
    cat3_ty_to[tyska[i]] = svenska[i]
    cat3_sv_to[svenska[i]] = tyska[i]


# Create category selection options depending on previous selection
cat1 = df.Cat_H1.values
cat1_unique = np.unique(cat1.astype(str))
cat2_dict = {}
for cat1_temp in cat1_unique:
    cat2_temp = df[df['Cat_H1']==cat1_temp].Cat_H2.values
    cat2_unique_temp = np.unique(cat2_temp.astype(str))
    cat2_dict[cat1_temp] = cat2_unique_temp

cat2 = df.Cat_H2.values
cat2_unique = np.unique(cat2.astype(str))
cat3_dict = {}
for cat2_temp in cat2_unique:
    cat3_temp = df[df['Cat_H2']==cat2_temp].Cat_H3.values
    cat3_unique_temp = np.unique(cat3_temp.astype(str))
    cat3_dict[cat2_temp] = cat3_unique_temp

# translate
#cat1_unique = [cat1_ty_to[cat1_unique[i]] for i in range(len(cat1_unique))]
cat1_unique = [cat1_ty_to[cat1_unique[i]] if cat1_unique[i] != 'nan' else 'nan' for i in range(len(cat1_unique))]

category1 = np.array([st.selectbox('Select category 1', cat1_unique, key='cat1')])
# german choice
category1 = np.array([cat1_sv_to[category1[0]]])
# translate cat 2
cat2_choices = cat2_dict[category1[0]] # in german
cat2_choices = [cat2_ty_to[cat2_choices[i]] if cat2_choices[i] != 'nan' else 'nan' for i in range(len(cat2_choices))]
category2 = np.array([st.selectbox('Select category 2', cat2_choices, key='cat2')])
# german choice
category2 = np.array([cat2_sv_to[category2[0]]])
# translate cat 3
cat3_choices = cat3_dict[category2[0]] # in german
cat3_choices = [cat3_ty_to[cat3_choices[i]] if cat3_choices[i] != 'nan' else 'nan' for i in range(len(cat3_choices))]
category3 = np.array([st.selectbox('Select category 3', cat3_choices, key='cat3')])
# german choice
category3 = np.array([cat3_sv_to[category3[0]]])
geo = np.array([st.selectbox('Select geography', geo_unique, key='geo')])

input_name = st.text_input('Write name of material:')

if input_name:
    text = [input_name]
    query = comb_bert(text, category1, category2, category3, geo)[0][0]
    print('query:', query)
    st.write('result:', query, 'kg CO2-eq per kg material')
