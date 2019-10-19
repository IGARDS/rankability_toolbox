# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.4'
#       jupytext_version: 1.2.4
#   kernelspec:
#     print_name: Python 3
#     language: python
#     name: python3
# ---

# # MovieLens Dataset example
# https://grouplens.org/datasets/movielens/

# %load_ext autoreload
# %autoreload 2
# %matplotlib inline

import sys
sys.path.append("/home/ubuntu/rankability_toolbox")

import pandas as pd
import numpy as np
import dill

import pyrankability

DATA_DIR='ml-latest-small'

# ## Read and preprocess the data

links = pd.read_csv("%s/links.csv"%DATA_DIR)
print(links.head())
links["tmdbId"] = links["tmdbId"].fillna(-1).astype(int)
print(links.head())
print(links.dtypes)

movies = pd.read_csv("%s/movies.csv"%DATA_DIR)
print(movies.head())
movies["genres"] = movies["genres"].str.split("|")
print(movies.head())

# +
from sklearn.preprocessing import MultiLabelBinarizer
mlb = MultiLabelBinarizer()

movies = movies.join(pd.DataFrame(mlb.fit_transform(movies['genres']),
                          columns=mlb.classes_,
                          index=movies.index))

# -

movies

ratings = pd.read_csv("%s/ratings.csv"%DATA_DIR)
print(ratings.head())

# ## Exploratory

# **Number of unique movies along with their counts**

ratings.set_index("movieId").join(movies.set_index("movieId"))["title"].value_counts()

# **What if we say 50 people in total had to rank a movie to even start considering it?**

counts = ratings.set_index("movieId").join(movies.set_index("movieId"))["title"].value_counts()
counts[counts > 50]

# **This gets us to 437 movies, so that's a decent D matrix I think.**
#
# For a start, let's fill in values in D by finding users who ranked two movies and then storing the difference in rating.

ratings.columns

# +
from itertools import combinations
flatten = lambda l: [item for sublist in l for item in sublist]
df1 = ratings.groupby("userId").apply(lambda df: 
                                     pd.DataFrame([flatten(tup) for tup in list(combinations(df.values,2))],columns=[v+"_i"for v in df.columns]+[v+"_j"for v in df.columns],dtype=int).
                                     set_index(["movieId_i","movieId_j"]))

df2 = df1.apply(lambda x: x["rating_i"]-x["rating_j"],axis=1).unstack()
print(df2)
#fix negatives when square
#inxs = np.where(df2 < 0)
#print("Fix negative numbers",len(inxs[0]))
#df2.values[inxs[1],inxs[0]] = -df2.values[inxs[0],inxs[1]]
#df2.values[inxs[0],inxs[1]] = 0
#print(df2.groupby().apply(lambda x: x.unstack()))
#inxs = np.where(df2 < 0)
#print("After fix negative numbers",len(inxs[0]))


# -

means = df2.stack().groupby(["movieId_i","movieId_j"]).mean()

counts = df2.stack().groupby(["movieId_i","movieId_j"]).count()

# **A quick glance at our counts and means**

print(counts.head())

print(counts.head())

# ## Sparse format to D
# We now have a sparse format, but we need to turn this into a D matrix. Let's only use a movie if there exists a paired count greater than 100.

count_mask = counts > 100
print(count_mask.sum())

# This means we would have a D matrix that is size 597 by 597. A little larger than our target of 500 x 500. But this is still small enough that we can construct the total matrix.

D_counts = counts[count_mask].unstack()
D_means = means[count_mask].unstack()

# We need to label the data so we can search for specific genres

D_counts_labelled = D_counts.join(movies.set_index("movieId")).transpose().join(movies.set_index("movieId"))
D_means_labelled = D_means.join(movies.set_index("movieId")).transpose().join(movies.set_index("movieId"))

# **Break the data into genres**

D_counts_labelled

genres = ["Comedy","Romance"]
genre = genres[1]

# +
from scipy.sparse import csr_matrix

i_mask = D_counts_labelled.loc[genre] == 1
j_mask = D_counts_labelled.loc[:,genre] == 1
D_means_genre = D_means.loc[i_mask,j_mask]
D_counts_genre = D_counts.loc[i_mask,j_mask]
inxs = np.where(~D_means_genre.isna())
n = max([max(inxs[0]),max(inxs[1])])+1
row = np.ix_(inxs[0],inxs[1])[0].reshape(1,-1)[0]
col = np.ix_(inxs[0],inxs[1])[1].reshape(1,-1)[0]
D = csr_matrix((D_means_genre.values[inxs].flatten(), (row,col)), shape=(n, n)).todense()
# Now fix negative numbers
inxs = np.where(D < 0)
print("Fix negative numbers",len(inxs[0]))
D[inxs[1],inxs[0]] = -D[inxs[0],inxs[1]]
D[inxs[0],inxs[1]] = 0
print("Just a subset of the matrix")
print(D[:10,:10])
inxs = np.where(D < 0)
print("After fix negative numbers",len(inxs[0]))
print("Summary of number of nonzero")
print(D.shape,(D>0).sum())
# -

print("hello")

# +
#k,details = pyrankability.hillside.count_lp(D)
#print(k,details["P"])

# +
#print(k,details["P"]) 

# +
#pyrankability.hillside.objective_count_exhaustive(D[:8,:8])
# -

D


