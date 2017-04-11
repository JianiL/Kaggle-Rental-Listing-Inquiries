
# coding: utf-8

import numpy as np
import pandas as pd
from itertools import product
import math
import random
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import StratifiedKFold
import csv
import reverse_geocoder as rg
from textblob import TextBlob

def add_features(df):
    fmt = lambda s: s.replace("\u00a0", "").strip().lower()
    # add new york city center information
    ny_lat = 40.785091
    ny_lon = -73.968285
    df["photo_count"] = df["photos"].apply(len)
    df["street_address"] = df['street_address'].apply(fmt)
    df["display_address"] = df["display_address"].apply(fmt)
    df["desc_wordcount"] = df["description"].apply(str.split).apply(len)
    df['feature_count'] = df["features"].apply(len)
    df["pricePerBed"] = df['price'] / df['bedrooms']
    df["pricePerBath"] = df['price'] / df['bathrooms']
    df["pricePerRoom"] = df['price'] / (df['bedrooms'] + df['bathrooms'])
    df["bedPerBath"] = df['bedrooms'] / df['bathrooms']
    df["bedBathDiff"] = df['bedrooms'] - df['bathrooms']
    df["bedBathSum"] = df["bedrooms"] + df['bathrooms']
    df["bedsPerc"] = df["bedrooms"] / (df['bedrooms'] + df['bathrooms'])
    df['bathPerc'] = df['bathrooms'] / (df['bedrooms'] + df['bathrooms'])
    df["distance"] = ((df['latitude'].astype(float) - ny_lat)**2 + (df['longitude'].astype(float) -ny_lon)**2)**(1/2.0)
    df['bedandBath'] = df['bedrooms'] + df['bathrooms']
    df['photoToRoom'] = df['photo_count'] / df['bedandBath']
    df["created"] = pd.to_datetime(df["created"])
    df["created_year"] = df["created"].dt.year
    df["created_month"] = df["created"].dt.month
    df["created_day"] = df["created"].dt.day
    df["east"] = df["street_address"].apply(lambda x: x.find('east')>-1).astype(int)
    df["west"] = df["street_address"].apply(lambda x: x.find('west')>-1).astype(int)
    #df["latlon"] = (df["latitude"]-df["longitude"]).astype('object')
    df["created"] = pd.to_datetime(df["created"])
    df["days_since"] = df["created"].max() - df["created"]
    df["days_since"] = (df["days_since"] / np.timedelta64(1, 'D')).astype(int)
    df["num_capital_letters"] = df["description"].apply(lambda x: sum(1 for c in x if c.isupper()))

    df = df.fillna(-1).replace(np.inf, -1)
    ls
    return df


def factorize(df1, df2, column):
    ps = df1[column].append(df2[column])
    factors = ps.factorize()[0]
    df1[column] = factors[:len(df1)]
    df2[column] = factors[len(df1):]
    return df1, df2


def designate_single_observations(df1, df2, column):
    ps = df1[column].append(df2[column])
    grouped = ps.groupby(ps).size().to_frame().rename(columns={0: "size"})
    df1.loc[df1.join(grouped, on=column, how="left")["size"] <= 1, column] = -1
    df2.loc[df2.join(grouped, on=column, how="left")["size"] <= 1, column] = -1
    return df1, df2


def hcc_encode(train_df, test_df, variable, target, prior_prob, k, f=1, g=1, r_k=None, update_df=None):
    """
    See "A Preprocessing Scheme for High-Cardinality Categorical Attributes in
    Classification and Prediction Problems" by Daniele Micci-Barreca
    """
    hcc_name = "_".join(["hcc", variable, target])

    grouped = train_df.groupby(variable)[target].agg({"size": "size", "mean": "mean"})
    grouped["lambda"] = 1 / (g + np.exp((k - grouped["size"]) / f))
    grouped[hcc_name] = grouped["lambda"] * grouped["mean"] + (1 - grouped["lambda"]) * prior_prob

    df = test_df[[variable]].join(grouped, on=variable, how="left")[hcc_name].fillna(prior_prob)
    if r_k: df *= np.random.uniform(1 - r_k, 1 + r_k, len(test_df))     # Add uniform noise. Not mentioned in original paper

    if update_df is None: update_df = test_df
    if hcc_name not in update_df.columns: update_df[hcc_name] = np.nan
    update_df.update(df)
    return


def create_binary_features(df):
    bows = {
        "dogs": ("dogs", "dog"),
        "cats": ("cats",),
        "nofee": ("no fee", "no-fee", "no  fee", "nofee", "no_fee"),
        "lowfee": ("reduced_fee", "low_fee", "reduced fee", "low fee"),
        "furnished": ("furnished",),
        "parquet": ("parquet", "hardwood"),
        "concierge": ("concierge", "doorman", "housekeep", "in_super"),
        "prewar": ("prewar", "pre_war", "pre war", "pre-war"),
        "laundry": ("laundry", "lndry"),
        "health": ("health", "gym", "fitness", "training"),
        "transport": ("train", "subway", "transport"),
        "parking": ("parking",),
        "utilities": ("utilities", "heat water", "water included")
    }

    def indicator(bow):
        return lambda s: int(any([x in s for x in bow]))

    features = df["features"].apply(lambda f: " ".join(f).lower())   # convert features to string
    for key in bows:
        df["feature_" + key] = features.apply(indicator(bows[key]))

    return df


# Load data
X_train = pd.read_json("train.json").sort_values(by="listing_id")
X_test = pd.read_json("test.json").sort_values(by="listing_id")


# Make target integer, one hot encoded, calculate target priors
X_train = X_train.replace({"interest_level": {"low": 0, "medium": 1, "high": 2}})
X_train = X_train.join(pd.get_dummies(X_train["interest_level"], prefix="pred").astype(int))
prior_0, prior_1, prior_2 = X_train[["pred_0", "pred_1", "pred_2"]].mean()

# Add common features
X_train = add_features(X_train)
X_test = add_features(X_test)

# Special designation for building_ids, manager_ids, display_address with only 1 observation
for col in ('building_id', 'manager_id', 'display_address'):
    X_train, X_test = designate_single_observations(X_train, X_test, col)

# High-Cardinality Categorical encoding
skf = StratifiedKFold(5)
attributes = product(("building_id", "manager_id"), zip(("pred_1", "pred_2"), (prior_1, prior_2)))
for variable, (target, prior) in attributes:
    hcc_encode(X_train, X_test, variable, target, prior, k=5, r_k=None)
    for train, test in skf.split(np.zeros(len(X_train)), X_train['interest_level']):
        hcc_encode(X_train.iloc[train], X_train.iloc[test], variable, target, prior, k=5, r_k=0.01, update_df=X_train)

# Create binarized features
X_train = create_binary_features(X_train)
X_test = create_binary_features(X_test)

def add_median_price(key=None, suffix="", trn_df=None, tst_df=None):
    # Set features to be used
    median_features = key[:]
    median_features.append('price')
    # Concat train and test to find median prices over whole dataset
    median_prices = pd.concat([trn_df[median_features], tst_df[median_features]], axis=0)
    # Group data by key to compute median prices
    medians_by_key = median_prices.groupby(by=key)['price'].median().reset_index()
    # Rename median column with provided suffix
    medians_by_key.rename(columns={'price': 'median_price_' + suffix}, inplace=True)
    # Update data frames
    trn_df = trn_df.merge(medians_by_key, on=key, how="left")
    tst_df = tst_df.merge(medians_by_key, on=key, how="left")
    trn_df['price_to_median_ratio_' + suffix] = trn_df['price'] / trn_df['median_price_' + suffix]
    tst_df['price_to_median_ratio_' + suffix] = tst_df['price'] / tst_df['median_price_' + suffix]

    return trn_df, tst_df

print('Adding price to median price ratio data')
X_train, X_test = add_median_price(key=["bedrooms"], suffix="bed", trn_df=X_train, tst_df=X_test)

# add the managre information
index=list(range(X_train.shape[0]))
random.shuffle(index)
a=[np.nan]*len(X_train)
b=[np.nan]*len(X_train)
c=[np.nan]*len(X_train)

for i in range(5):
    building_level={}
    for j in X_train['manager_id'].values:
        building_level[j]=[0,0,0]
    test_index=index[int((i*X_train.shape[0])/5):int(((i+1)*X_train.shape[0])/5)]
    train_index=list(set(index).difference(test_index))
    for j in train_index:
        temp=X_train.iloc[j]
        if temp['interest_level']== 0:
            building_level[temp['manager_id']][0]+=1
        if temp['interest_level']== 1:
            building_level[temp['manager_id']][1]+=1
        if temp['interest_level']== 2:
            building_level[temp['manager_id']][2]+=1
    for j in test_index:
        temp=X_train.iloc[j]
        if sum(building_level[temp['manager_id']])!=0:
            a[j]=building_level[temp['manager_id']][0]*1.0#/sum(building_level[temp['manager_id']])
            b[j]=building_level[temp['manager_id']][1]*1.0#/sum(building_level[temp['manager_id']])
            c[j]=building_level[temp['manager_id']][2]*1.0#/sum(building_level[temp['manager_id']])
X_train['manager_level_low']=a
X_train['manager_level_medium']=b
X_train['manager_level_high']=c



a=[]
b=[]
c=[]
building_level={}
for j in X_train['manager_id'].values:
    building_level[j]=[0,0,0]
for j in range(X_train.shape[0]):
    temp=X_train.iloc[j]
    if temp['interest_level']== 0:
        building_level[temp['manager_id']][0]+=1
    if temp['interest_level']== 1:
        building_level[temp['manager_id']][1]+=1
    if temp['interest_level']== 2:
        building_level[temp['manager_id']][2]+=1

for i in X_test['manager_id'].values:
    if i not in building_level.keys():
        a.append(np.nan)
        b.append(np.nan)
        c.append(np.nan)
    else:
        a.append(building_level[i][0]*1.0)#/sum(building_level[i]))
        b.append(building_level[i][1]*1.0)#/sum(building_level[i]))
        c.append(building_level[i][2]*1.0)#/sum(building_level[i]))
X_test['manager_level_low']=a
X_test['manager_level_medium']=b
X_test['manager_level_high']=c


# Factorize building_id, display_address, manager_id, street_address
for col in ( 'display_address', 'manager_id', 'street_address', 'building_id'):
    X_train, X_test = factorize(X_train, X_test, col)



# add the building id to differnet level
X_train['flag'] = 'train'
X_test['flag'] = 'test'
full_data = pd.concat([X_train, X_test], axis=0)

buildings_count = full_data['building_id'].value_counts()

full_data['top_10_building'] = full_data['building_id'].apply(lambda x: 1 if x in buildings_count.index.values[
    buildings_count.values >= np.percentile(buildings_count.values, 90)] else 0)
full_data['top_25_building'] = full_data['building_id'].apply(lambda x: 1 if x in buildings_count.index.values[
    buildings_count.values >= np.percentile(buildings_count.values, 75)] else 0)
full_data['top_5_building'] = full_data['building_id'].apply(lambda x: 1 if x in buildings_count.index.values[
    buildings_count.values >= np.percentile(buildings_count.values, 95)] else 0)
full_data['top_50_building'] = full_data['building_id'].apply(lambda x: 1 if x in buildings_count.index.values[
    buildings_count.values >= np.percentile(buildings_count.values, 50)] else 0)
full_data['top_1_building'] = full_data['building_id'].apply(lambda x: 1 if x in buildings_count.index.values[
    buildings_count.values >= np.percentile(buildings_count.values, 99)] else 0)
full_data['top_2_building'] = full_data['building_id'].apply(lambda x: 1 if x in buildings_count.index.values[
    buildings_count.values >= np.percentile(buildings_count.values, 98)] else 0)
full_data['top_15_building'] = full_data['building_id'].apply(lambda x: 1 if x in buildings_count.index.values[
    buildings_count.values >= np.percentile(buildings_count.values, 85)] else 0)
full_data['top_20_building'] = full_data['building_id'].apply(lambda x: 1 if x in buildings_count.index.values[
    buildings_count.values >= np.percentile(buildings_count.values, 80)] else 0)
full_data['top_30_building'] = full_data['building_id'].apply(lambda x: 1 if x in buildings_count.index.values[
    buildings_count.values >= np.percentile(buildings_count.values, 70)] else 0)


full_coords = full_data[["listing_id", "latitude", "longitude"]]

for i, j in full_data.iterrows():
    lat_lon.append((j["latitude"], j["longitude"]))
    listings.append(int(j["listing_id"]))

from nltk.stem import PorterStemmer
import re
# Removes symbols, numbers and stem the words to reduce dimentional space
stemmer = PorterStemmer()
def clean(x):
    regex = re.compile('[^a-zA-Z]')
    i = regex.sub(' ', x).lower()
    i = i.split(" ")
    i = [stemmer.stem(l) for l in i]
    i = " ".join([l.strip() for l in i if (len(l) > 2)])
    return i

full_data['description_new'] = full_data['description'].apply(lambda x: clean(x))

# get the sentiment from the description

full_data['polarity'] = full_data['description'].apply(lambda x: TextBlob(x).sentiment.polarity)

full_data['subjectivity'] = full_data['description'].apply(lambda x: TextBlob(x).sentiment.subjectivity)


from sklearn.feature_extraction.text import CountVectorizer

cvert_desc = CountVectorizer(stop_words='english', max_features=200)
full_sparse = cvert_desc.fit_transform(full_data['description_new'])
col_desc = ['desc_' + i for i in cvert_desc.get_feature_names()]
count_vect_df = pd.DataFrame(full_sparse.todense(), columns = col_desc)
full_data = pd.concat([full_data.reset_index(), count_vect_df], axis = 1)


X_train = (full_data[full_data.flag=='train'])
X_test = (full_data[full_data.flag=='test'])
# Save
X_train = X_train.sort_index(axis=1).sort_values(by="listing_id")
X_test = X_test.sort_index(axis=1).sort_values(by="listing_id")
columns_to_drop = ["photos", "pred_0","pred_1", "pred_2", "description", "features", "created", 'flag', "description_new",
                  'building_id', 'street_address', 'sentiment']
X_train.drop(columns_to_drop, axis=1, errors="ignore", inplace = True)
X_test.drop(columns_to_drop, axis=1, errors="ignore", inplace = True)

# EDA on the sentiment
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style = 'whitegrid', color_codes = True)
sns.set(font_scale=2)
order = ['high', 'medium', 'low']
sns.boxplot(y, X_train['polarity'])
plt.title('test')
plt.show()

import xgboost as xgb
param = {}
param['objective'] = 'multi:softprob'
param['eta'] = 0.025
param['max_depth'] = 6
param['silent'] = 1
param['num_class'] = 3
param['eval_metric'] = "mlogloss"
param['min_child_weight'] = 1
param['subsample'] = 0.7
param['colsample_bytree'] = 0.7
param['nthread'] = 8
num_rounds = 1000


y = X_train.pop('interest_level')
X_test.drop(['interest_level'], axis = 1, inplace = True)


import xgboost as xgb
xgb_param = {'silent' : 1, 'eta': 0.025, 'max_depth':6, 'objective': 'multi:softprob',
             'eval_metric': 'mlogloss', 'subsample': 0.7, 'num_class': 3, 'min_child_weight': 2,
             'colsample_bytree': 0.7}
# Train on full data
dtrain = xgb.DMatrix(X_train,y)
dtest = xgb.DMatrix(X_test)
clf = xgb.train(xgb_param, dtrain, 1500, evals=([dtrain,'train'], [dtrain,'train']))
pred = clf.predict(dtest)
print("Saving Results.")
preds = pd.DataFrame({"listing_id": np.array(X_test['listing_id']), "high": pred[:,2], "medium": pred[:,1], "low": pred[:,0]})
preds = preds[["listing_id", "high", "medium", "low"]]
preds.to_csv('xgb_withsentiment' + '.csv', index=False)
