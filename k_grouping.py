from pyclustering.cluster.kmedoids import kmedoids
from pyclustering.cluster.elbow import elbow
import numpy as np
import json
import time

print('Successfully loaded all modules')

# load image encodings
print('Loading encodings...')
start = time.time()
encodings = "encodings.npy"
encodings = np.load(encodings)
stop = time.time()
print('Encodings loaded successfully', '[', round(stop-start, 2), 'seconds ]')

# create elbow instance
print('Creating elbow instance...')
start = time.time()
elbow_instance = elbow(encodings, 2, 100)
stop = time.time()
print('Elbow instance created', '[', round(stop-start, 2 ), 'seconds ]')

# find the optimal value for K (no. of groups) using elbow algorithm
print('Getting the optimal number of clusters using elbow...')
start = time.time()
elbow_instance.process()
K = elbow_instance.get_amount()
stop = time.time()
print(K, 'clusters should be formed according to Elbow', '[', round(stop-start, 2), 'seconds ]')

# load the distance matrix (similarity matrix)
print('Loading the similarity (distance) matrix...')
start = time.time()
distance_matrix = "cosine_similarity_matrix.npy"
distance_matrix = np.load(distance_matrix)
stop = time.time()
print('Similarity matrix loaded', '[', round(stop-start, 2), 'seconds ]')

# define K initial medoids randomly
print('Choosing', K, 'initial medoids randomly...')
start = time.time()
initial_medoids = [int(np.random.uniform(0, distance_matrix.shape[0])) for i in range(K)]
stop = time.time()
print('Random medoids selected', '[', round(stop-start, 2), 'seconds ]')
print('Random medoids are', initial_medoids)

# execute the K-Medoids algorithm
print('Creating Kmediod instance...')
start = time.time()
kmedoids_instance = kmedoids(distance_matrix, initial_medoids, data_type='distance_matrix')
stop = time.time()
print('Created Kmedoid instance', '[', round(stop-start, 2), 'seconds ]')

# get actual K medoids and clusters
print('Get clusters and medoids using K-Medoids algorithm...')
start = time.time()
kmedoids_instance.process()
clusters = kmedoids_instance.get_clusters()
medoids = kmedoids_instance.get_medoids()
stop = time.time()
print('Found clusters and medoids', '[', round(stop-start, 2), 'seconds ]')

print('Final medoids are', medoids)

# write the final medoids and clusters to a JSON file
print('Writing medoids and clusters to JSON file...')
start = time.time()
with open('k_groups.json', 'w') as file:
    json.dump([clusters, medoids], file, indent=4)
stop = time.time()
print('Writing completed', '[', round(stop-start, 2), 'seconds ]')

print('All tasks completed. Exiting...')
