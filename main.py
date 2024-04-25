"""Sentiment Analysis using OHE for word embedding"""

"""Packages"""
import pandas as pd
from collections import Counter
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from dataset_class import SentimentData,  SentimentModel
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch
import seaborn as sns
import matplotlib.pyplot as plt

pd.options.display.width = 0
"""Importing"""
df = pd.read_csv('tweets.csv', encoding='ISO-8859-1').dropna()


# print(df['sentiment'])
# %% get class values based on categories
cat_id = {'neutral': 1,
          'negative': 0,
          'positive': 2}

df['class'] = df['sentiment'].map(cat_id)

#%% Hyperparameters
BATCH_SIZE = 512
NUM_EPOCHS = 80

#%% separate independent and dependent features
X = df['text'].values
y = df['class'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.5, random_state=123)
print(f"X train: {X_train.shape}, y train: {y_train.shape}\nX test: {X_test.shape}, y test: {y_test.shape}")

one_hot = CountVectorizer()
X_train_onehot = one_hot.fit_transform(X_train)
X_test_onehot = one_hot.transform(X_test)


#Create a dataset

train_ds = SentimentData(X= X_train_onehot, y = y_train)
test_ds = SentimentData(X_test_onehot, y_test)

# %% Dataloader
train_loader = DataLoader(dataset=train_ds, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_ds, batch_size=15000)



#%% Model, Loss and Optimizer
model = SentimentModel(NUM_FEATURES = X_train_onehot.shape[1], NUM_CLASSES = 3)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters())

# %% Model Training

train_losses = []
for e in range(NUM_EPOCHS):
    curr_loss = 0
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        y_pred_log = model(X_batch)
        loss = criterion(y_pred_log, y_batch.long())

        curr_loss += loss.item()
        loss.backward()
        optimizer.step()
    train_losses.append(curr_loss)
    print(f"Epoch {e}, Loss: {curr_loss}")

# %%
sns.lineplot(x=list(range(len(train_losses))), y=train_losses)
plt.show()
# %% Model Evaluation
with torch.no_grad():
    for X_batch, y_batch in test_loader:
        y_test_pred_log = model(X_batch)
        y_test_pred = torch.argmax(y_test_pred_log, dim=1)

# %%
y_test_pred_np = y_test_pred.squeeze().cpu().numpy()

# %%
acc = accuracy_score(y_pred=y_test_pred_np, y_true=y_test)
f"The accuracy of the model is {np.round(acc, 3) * 100}%."
# %%
# most_common_cnt = Counter(y_test).most_common()[0][1]
# print(f"Naive Classifier: {np.round(most_common_cnt / len(y_test) * 100, 1)} %")
# # %% Confusion Matrix
sns.heatmap(confusion_matrix(y_test_pred_np, y_test), annot=True, fmt=".0f")
plt.show()
# %%