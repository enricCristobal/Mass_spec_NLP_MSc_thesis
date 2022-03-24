#!/usr/bin/env python

import pickle
import os
import numpy as np
import matplotlib.pyplot as plt    

def get_inten_dist(directory):
    intensity_array = []; intensity_zoom = []; large_int_count = 0
    for i,file in enumerate(os.listdir(directory)):
        if i < 3:
            x = pickle.load(open(directory + file, "rb"))
            file_intensities = []
            for i in range(len(x)):
                #print('line', i)
                mz_array = x.iloc[i,5]
                intensities = x.iloc[i,6]
                #if mz_array[0] < 900:
                #    file_intensities.append(intensities)
                for element in intensities:
                    if element > 1e7:
                        large_int_count += 1
                        continue
                    else:
                        if element > 1e4:
                            intensity_array.append(element)
                        else:
                            intensity_zoom.append(element)
                        #print('Writing!')
                        #file_intensities.append(element)
                            intensity_array.append(element)
            #print('Inserting!')
            #flat_list = [int(element) for line in file_intensities for element in line]
            #intensity_array.append(file_intensities)
    #return file_intensities
    return intensity_array, intensity_zoom, large_int_count

dir = os.getcwd() + '\\pickle_data\\'
intensities, intensities_zoom, counts = get_inten_dist(dir)
print('Large intensity counts 1e7 : ', counts)
#flat_list = [int(element) for line in intensities for element in line]
#print(intensities[0][0:10])

plot1 = plt.figure(1)
plt.hist(intensities, bins = 100)
plot2 = plt.figure(2)
plt.hist(intensities_zoom, bins = 100)
plt.show()


