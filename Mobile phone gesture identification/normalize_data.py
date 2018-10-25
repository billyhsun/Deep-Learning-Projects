'''
    Normalize the data, save as ./data/normalized_data.npy
'''

import numpy as np


def normalize(array):
    # First option: local mean and standard deviation
    new_arr = np.zeros([array.shape[0], 6, 100])
    for i in range(array.shape[0]):
        temp = np.transpose(array[i])
        for k in range(6):
            loc_mean = np.mean(temp[k])
            loc_std = np.std(temp[k])
            for l in range(100):
                temp[k, l] = (temp[k, l] - loc_mean)/loc_std
        new_arr[i] = temp

    # Second option: global mean and standard deviation
    '''
    glb_means = [0, 0, 0, 0, 0, 0]
    glb_std = [0, 0, 0, 0, 0, 0]
    for i in range(5590):
        temp = np.transpose(array[i])
        for j in range(6):
            glb_means[j] += np.sum(temp[j])

    for x in range(6):
        glb_means[x] = glb_means[x]/559000
        
    for i in range(5590):
        temp = np.transpose(array[i])
        for k in range(6):
            for l in range(100):
                glb_std[k] += ((temp[k, l] - glb_means[k])**2)

    # for y in range(6):
    #    glb_std[y] = math.sqrt(glb_means[y])
    
    new_arr = np.zeros([5590, 6, 100])
    for i in range(5590):
        temp = np.transpose(array[i])
        for k in range(6):
            for l in range(100):
                temp[k, l] = (temp[k, l] - glb_means[k])/glb_std[k]
        new_arr[i] = temp
    '''
    # np.save('normalized_data.npy', new_arr)
    # print(new_arr)
    return new_arr
