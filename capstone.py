#!/usr/bin/env python
# coding: utf-8

# # Djentbox Song Recommendation

# In[1]:


#<----------CELL 1---------->

import pandas as pd

data = pd.read_csv('music_metadata.csv')
data = data.dropna()

features = data[['Danceability', 'Energy', 'Speechiness', 'Acousticness', 'Instrumentalness', 'Liveness', 'Valence']]

num_samples, num_features = features.shape
print("Samples: {}, Features: {}".format(num_samples, num_features))


# In[2]:


#<----------CELL 2---------->

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt

reduced_data = PCA(n_components=2).fit_transform(features)
kmeans = KMeans(init="k-means++", n_clusters=5, n_init=4)
kmeans.fit(reduced_data)

h = 0.02

x_min, x_max = reduced_data[:, 0].min() - 1, reduced_data[:, 0].max() + 1
y_min, y_max = reduced_data[:, 1].min() - 1, reduced_data[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])

z = z.reshape(xx.shape)
plt.figure(1)
plt.clf() 
plt.imshow(
    z,
    interpolation="nearest",
    extent=(xx.min(), xx.max(), yy.min(), yy.max()),
    cmap=plt.cm.Paired,
    aspect="auto",
    origin="lower",
)

plt.plot(reduced_data[:, 0], reduced_data[:, 1], "k.", markersize=2)

centroids = kmeans.cluster_centers_
plt.scatter(
    centroids[:, 0],
    centroids[:, 1],
    marker="x",
    s=169,
    linewidths=3,
    color="w",
    zorder=10,
)
plt.title(
    "K-means clustering (PCA reduced data)\n"
    "Centroids marked with white cross"
)
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.xticks(())
plt.yticks(())
plt.show()


# In[3]:


#<----------CELL 3---------->

danceability_data = features['Danceability'].tolist()
plt.hist(danceability_data)
plt.title("Danceability feature value variation")
plt.show()


# In[4]:


#<----------CELL 4---------->

energy_data = features['Energy'].tolist()
speechiness_data = features['Speechiness'].tolist()
acousticness_data = features['Acousticness'].tolist()
instrumentalness_data = features['Instrumentalness'].tolist()
liveness_data = features['Liveness'].tolist()
valence_data = features['Valence'].tolist()
all_data = [danceability_data, energy_data, speechiness_data, acousticness_data, instrumentalness_data, liveness_data,
           valence_data]

plt.boxplot(all_data)
plt.show()


# In[5]:


#<----------CELL 5---------->

import ipywidgets as widgets
from IPython.display import display

track_names = data['Track'].tolist()

song_name_input = widgets.Text(placeholder='Enter song name...')

suggestion_dropdown = widgets.Dropdown(
    options = [],
    disabled=True
)

def update_suggestions(change):
    user_input = change['new'].lower()
    suggestions = [track for track in track_names if track.lower().startswith(user_input)]
    suggestion_dropdown.options = suggestions[:10]
    suggestion_dropdown.disabled = not bool(user_input)

def handle_dropdown_selection(change):
    selected_song = change['new']
    if selected_song:
        song_name_input.value = selected_song
        clear_output()
        display(song_name_input, suggestion_dropdown)

song_name_input.observe(update_suggestions, names='value')
suggestion_dropdown.observe(handle_dropdown_selection, names='value')

def get_selected_song():
    return suggestion_dropdown.value

print('Start typing a song name and then select complete song name from dropdown!')
print('(The catalog is large!)')
print()

display(song_name_input, suggestion_dropdown)


# In[6]:


#<----------CELL 6---------->

kmeans = KMeans(init="k-means++", n_clusters=5, n_init=4)
kmeans.fit(features)
predictions = kmeans.predict(features)
cluster_distances = kmeans.transform(features)
reference_song = get_selected_song()
reference_song_object = 0

class Song:
    def __init__(self, artist, song, album, danceability, energy, speechiness, acousticness, instrumentalness, liveness, valence,
                 cluster, cluster_distance):
        self.artist = artist
        self.song = song
        self.album = album
        self.danceability = danceability
        self.energy = energy
        self.speechiness = speechiness
        self.acousticness = acousticness
        self.instrumentalness = instrumentalness
        self.liveness = liveness
        self.valence = valence
        self.cluster = cluster
        self.cluster_distance = cluster_distance

songs = []
for index, prediction in enumerate(predictions):
    artist = data.iloc[index]['Artist']
    song = data.iloc[index]['Track']
    album = data.iloc[index]['Album']
    danceability = data.iloc[index]['Danceability']
    energy = data.iloc[index]['Energy']
    speechiness = data.iloc[index]['Speechiness']
    acousticness = data.iloc[index]['Acousticness']
    instrumentalness = data.iloc[index]['Instrumentalness']
    liveness = data.iloc[index]['Liveness']
    valence = data.iloc[index]['Valence']
    cluster = prediction
    cluster_distance = cluster_distances[index][cluster]
    
    song = Song(artist, song, album, danceability, energy, speechiness, acousticness, instrumentalness, liveness, valence,
                cluster, cluster_distance)
    songs.append(song)

for song in songs:
    if song.song == reference_song:
        reference_song_object = song
        break

songs_in_reference_cluster = []
for song in songs:
    if song.cluster == reference_song_object.cluster:
        songs_in_reference_cluster.append(song)

songs_in_reference_cluster.remove(reference_song_object)

song_with_smallest_distance = 0
smallest_distance = 999
for song in songs_in_reference_cluster:
    if abs(reference_song_object.cluster_distance - song.cluster_distance) < smallest_distance:
        smallest_distance = abs(reference_song_object.cluster_distance - song.cluster_distance)
        song_with_smallest_distance = song

print("Based on your song choice, the Djentbox recommendation is:")
print(song_with_smallest_distance.song, end="")
print(" by {}".format(song_with_smallest_distance.artist))


# In[7]:


#<----------CELL 7---------->

print("Selection: ")
print("--- Artist: {}".format(reference_song_object.artist))
print("--- Song: {}".format(reference_song_object.song))
print("--- Album: {}".format(reference_song_object.album))
print("--- Danceability: {}".format(reference_song_object.danceability))
print("--- Energy: {}".format(reference_song_object.energy))
print("--- Speechiness: {}".format(reference_song_object.speechiness))
print("--- Acousticness: {}".format(reference_song_object.acousticness))
print("--- Instrumentalness: {}".format(reference_song_object.instrumentalness))
print("--- Liveness: {}".format(reference_song_object.liveness))
print("--- Valence: {}".format(reference_song_object.valence))

print()

print("Recommendation: ")
print("--- Artist: {}".format(song_with_smallest_distance.artist))
print("--- Song: {}".format(song_with_smallest_distance.song))
print("--- Album: {}".format(song_with_smallest_distance.album))
print("--- Danceability: {}".format(song_with_smallest_distance.danceability))
print("--- Energy: {}".format(song_with_smallest_distance.energy))
print("--- Speechiness: {}".format(song_with_smallest_distance.speechiness))
print("--- Acousticness: {}".format(song_with_smallest_distance.acousticness))
print("--- Instrumentalness: {}".format(song_with_smallest_distance.instrumentalness))
print("--- Liveness: {}".format(song_with_smallest_distance.liveness))
print("--- Valence: {}".format(song_with_smallest_distance.valence))

print()

print("Both songs are in same k-means cluster no. {}".format(song_with_smallest_distance.cluster))
print("Selected song distance from cluster center: {}".format(reference_song_object.cluster_distance))
print("Recommended song distance from cluster center: {}".format(song_with_smallest_distance.cluster_distance))
print("Positional difference from cluster center: {}".format(abs(reference_song_object.cluster_distance - song_with_smallest_distance.cluster_distance)))


# In[ ]:




