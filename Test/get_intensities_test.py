#!/usr/bin/env python

import pickle
import os
import numpy as np
import matplotlib.pyplot as plt    

def get_inten_dist(directory, high_int, low_int, percentiles_array):
    intensity_array = []; intensity_zoom = []; large_int_count = 0
    for i,file in enumerate(os.listdir(directory)):
        if i < 3:
            print(i)
            x = pickle.load(open(directory + file, "rb"))
            x = x.iloc[9::3, [5,6]] # get rid of the "algorithm warmup" and "magic" scans
            file_intensities = []
            for i in range(len(x)):
                #print('line', i)
                mz_array = x.iloc[i,0]
                intensities = x.iloc[i,1]
                #if mz_array[0] < 900:
                #    file_intensities.append(intensities)
                for element in intensities:
                    if element > high_int:
                        large_int_count += 1
                        continue
                    else:
                        if element > low_int:
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
    percentiles = np.percentile(intensity_array, percentiles_array)
    percentiles = [round(num, 2) for num in percentiles]
    return intensity_array, intensity_zoom, percentiles, large_int_count

dir = os.getcwd() + '\\pickle_data\\'
high_int = 1e7
low_int = 1e5
percentiles_array = [25, 50, 75, 90]
intensities, intensities_zoom, percentiles, counts = get_inten_dist(dir, high_int, low_int, percentiles_array)
print(f'Intensity counts larger than {high_int:e} : {counts}')
print(f'{percentiles_array} percentiles = {percentiles}')
#flat_list = [int(element) for line in intensities for element in line]
#print(intensities[0][0:10])

plot1 = plt.figure(1)
plt.hist(intensities, bins = 100)
plt.savefig('testing/Intensity_big.png')
plot2 = plt.figure(2)
plt.hist(intensities_zoom, bins = 100)
plt.savefig('testing/Inten_small.png')



