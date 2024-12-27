import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random

#%%
# Generate sample of random integers from 1 to 10 including
random.seed(3101)
sample_1 = np.random.randint(11, size=100000)

#%%
len(sample_1)

#%%
# To avoid an unusual peak, set bins=11 to match the range of the data (0–10 inclusive)
fig_1 = plt.hist(sample_1, bins=11)
plt.show()


#%%
# Generate sample of random float numbers from 0 to 1
random.seed(3101)
sample_2 = np.random.uniform(0, 1, size=100000)

#%%
# Get the last digit of the float part of a number
sample_2_ult = [int(str(x)[-1]) for x in sample_2]

#%%
# Show the histogram
fig_2 = plt.hist(sample_2_ult, bins=9)
plt.show()

#%%
# Combine the array and the list into a single 1D structure
combined = np.concatenate([sample_1, sample_2_ult])

#%%
# Create a dataframe of the combined data
df = pd.DataFrame(combined, columns=['digits'])
len(df)

#%%
# Create subplots()
fig, axes = plt.subplots(1,2, figsize=(12,8))

# First subplot: Histogram of the digits
axes[0].hist(df['digits'], bins=11, color='blue')
axes[0].set_title('Histogram')
axes[0].set_xlabel('Digits')
axes[0].set_ylabel('Frequency')

# Second subplot: Boxplot of the digits
axes[1].boxplot(df['digits'], vert=False, patch_artist=True)
axes[1].set_title('Boxplot')
axes[1].set_xlabel('Digits')

# Show subplots
plt.tight_layout()
plt.show()


#%%
# Get the penultimate digit of the float part of a number
sample_2_penult = [int(str(x)[-2]) for x in sample_2]

#%%
# Create a dataframe
df_2 = pd.DataFrame(sample_2_penult, columns=['digits'])

#%%
# Divide the data into 5 groups
bins = [0, 2, 4, 6, 8, 10]
labels = ['0-1', '2-3', '4-5', '6-7', '8-9']
df_2['group'] = pd.cut(df_2['digits'], bins=bins, labels=labels, right=False, include_lowest=True)

#%%
# Check the distribution of values per group
df_2['group'].value_counts()

#%%
# Show histogram with the distribution of values per group
fig_3 = plt.hist(df_2['group'], bins=5)
plt.show()