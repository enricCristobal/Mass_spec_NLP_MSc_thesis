#!/usr/bin/env python

import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
import scipy.stats as stats

# Define directories where we get the spectra and where we write our translation to our vocab
scans_dir = '/home/projects/cpr_10006/projects/gala_ald/data/plasma_scans/ms1_scans/'

def token_counter(input_dir, peak_filter:float):

    count_vector = []
    for filename in os.listdir(input_dir): 
        if filename != '20190525_QE10_Evosep1_P0000005_LiNi_SA_Plate3_A7.raw.pkl': # Giving issues after manual inspection
            x= pickle.load(open(input_dir + filename, "rb"))
            x = x.iloc[9::3, [5,6]] # we get rid of the "algorithm warmup" and "magic" scans (obtained when visualizing df obtained from unpickle_to_df)
            for i in range(len(x)):
                mz_array = x.iloc[i,0]
                inten_array = x.iloc[i,1]

                # skip if it starts with 900 m/z as this is an artefact in the machine
                if mz_array[0] < 900:
                    # masking out the lower peaks' proportion ( !!TODO: at some point calculate mass over charge ratio)
                    if peak_filter != 1:
                        threshold = np.quantile(inten_array, np.array([1-peak_filter]))[0]
                        mask = np.ma.masked_where(inten_array<threshold, inten_array) # mask bottom 90% peaks that are below the threshold
                        mz_masked_array = np.ma.masked_where(np.ma.getmask(mask), mz_array)
                        mz_flt_array = mz_array[mz_masked_array.mask==False]
                    else:
                        mz_flt_array = mz_array

                # Collect
                count_vector.append(len(mz_flt_array))
    return count_vector


count_vector = token_counter (input_dir=scans_dir, peak_filter=0.0125)
mean = np.mean(count_vector)
sd = np.std(count_vector)
x = np.linspace(mean - 3*sd, mean + 3*sd, 100)
plt.hist(count_vector, bins = 50, density = True, alpha = 0.8)

plt.plot(x, stats.norm.pdf(x, mean, sd), linewidth=3)
plt.axvline(mean, color='red', linewidth = 1)
plt.title('Sentence length distribution (mean=%5d, std≈%5d)' %(mean, sd))
plt.savefig('/home/projects/cpr_10006/people/enrcop/Figures/sen_len_hist/sen_len_count_filter_0.0125.png')
#plt.show()
