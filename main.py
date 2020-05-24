import pandas as pd
import networkx as nx

# Ignore matplotlib warnings
import warnings
from operator import itemgetter
from networkx.algorithms import community
from sklearn.manifold import TSNE
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from node2vec import Node2Vec
sns.set_style('whitegrid')

# create dataframe
df = pd.read_csv("data_caleg.csv")
df[['dapil','nama','partai','lokasi']].head()

# create graph from nama to dapil
g = nx.from_pandas_edgelist(df, source='nama', target='dapil') 

# create new dataframe for labels
df2= df[['nama', 'partai']].dropna(axis = 0, how ='any') 
mylist = df2.values.tolist()

# append label  to edge
g.add_edges_from(mylist,label="partai")

# visualization before link prediction
labels = [i for i in dict(g.nodes).keys()]
labels = {i:i for i in dict(g.nodes).keys()}
labels

fig, ax = plt.subplots(figsize=(46,23))
layout = nx.spring_layout(g,iterations=50)
nx.draw_networkx_nodes(g, layout, ax = ax, labels=False)
nx.draw_networkx_edges(g, layout,  ax=ax, labels=False)

caleg = [person for person in df.nama if g.degree(person) > 1]
nx.draw_networkx_nodes(g, layout, nodelist=caleg, node_color='orange', node_size=400)

# jaccard algorithm
preds = nx.jaccard_coefficient(g)
count = 0
new_nodes = []
for u, v, p in preds:
    if(p > 0.9):
        count+= 1
        dicts = {'caleg1' : u,'caleg2': v}
        new_nodes.append(dicts)
nl = pd.DataFrame(new_nodes)
# save to csv if needed to easier importingdata
# nl.to_csv('directory/jaccard.csv',index=False)

# adamic-adar index algorithm

preds = nx.adamic_adar_index(g)
count = 0
new_nodes2 = []
for u, v, p in preds:
    if(p > 0.4):
        count +=1
        dicts2 = {'caleg1' : u,'caleg2': v, 'score' : p}
        new_nodes2.append(dicts2)
        if(count > 500000):
            break
nl2 = pd.DataFrame(new_nodes2)
# save to csv if needed to easier importingdata
# nl2.to_csv('directory/adamic.csv',index=False)

# preferential algorithm
preds = nx.preferential_attachment(g)
count = 0
new_nodes3 = []
for u, v, p in preds:
    count+=1
    dicts3 = {'caleg1' : u,'caleg2': v, 'score' : p}
    new_nodes3.append(dicts3)
    if(count > 20000000):
        break
nl3 = pd.DataFrame(new_nodes3)
nl3.head()
# save to csv if needed to easier importingdata
# nl3.to_csv('directory/preferential.csv',index=False)

# select random 50 from jaccard nodes 
df_new = pd.read_csv("directory/jaccard.csv")
df_neww = df_new.drop_duplicates(subset = ["caleg1"])
df_neww.head()

df_rand = df_neww.sample(50)
nl = []
for i in df_rand['caleg1']:
    nl.append(i)
    
final_df = df[df['nama'].isin(nl)]
new_df = final_df.drop_duplicates(subset = ["nama"])
g = nx.from_pandas_edgelist(new_df, source='nama', target='dapil') 
new_df2= new_df[['nama', 'partai']].dropna(axis = 0, how ='any') 
mylist = new_df2.values.tolist()
g.add_edges_from(mylist,label="partai")
labels = [i for i in dict(g.nodes).keys()]
labels = {i:i for i in dict(g.nodes).keys()}
preds = nx.adamic_adar_index(g)

# preferential algorithm
# preds = nx.preferential_attachment(g)
count = 0
new_nodes = []
for u, v, p in preds:
    if(p > 0):
        count+= 1
        dicts = {'caleg1' : u,'caleg2': v, 'score' : p}
        new_nodes.append(dicts)
nl = pd.DataFrame(new_nodes)
#new visualization
fig, ax = plt.subplots(figsize=(30,15))
layout = nx.spring_layout(g,iterations=50)
nx.draw_networkx_nodes(g, layout, ax = ax, labels=True)
nx.draw_networkx_edges(g, layout,  ax=ax)
_ = nx.draw_networkx_labels(g, layout, labels, ax=ax)
caleg = [person for person in new_df.nama if g.degree(person) > 1]
nx.draw_networkx_nodes(g, layout, nodelist=caleg, node_color='orange', node_size=150)