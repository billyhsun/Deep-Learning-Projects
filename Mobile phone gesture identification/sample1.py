import numpy as np
import os
from sklearn.preprocessing import LabelEncoder, OneHotEncoder


instance_matrix = np.load('instances.npy')
label_matrix = np.load('labels.npy')

print(np.shape(instance_matrix))
vec_norm = np.empty((5590, 100, 6))
for i in range(len(label_matrix)):
    sample_vec = instance_matrix[i,:,:]
    values_i = sample_vec
    mean_i = np.mean(values_i, axis=0)
    std_i = np.std(values_i, axis=0)
    for j in range(np.shape(values_i)[0]):
        v = np.array([np.divide((values_i[j] - mean_i), (std_i))])
        vec_norm[i, j] = v



#At this point we have concatenated the 215x100x6 all together to make one big matrix 5590x100x6 for all the gestures


np.save("./data/normalized_data.npy", vec_norm)
print(np.shape(vec_norm))



label_encoder = LabelEncoder()
one_h_encoder = OneHotEncoder()

label_matrix= np.reshape(label_matrix, 5590)
new_labels = label_encoder.fit_transform(label_matrix)
#new_labels = one_h_encoder.fit_transform(new_labels.reshape(-1, 1))
#print(np.shape(new_labels))
np.save("./data/encoded_labels.npy",new_labels)