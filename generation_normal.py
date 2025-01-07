#%%
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_curve, auc
import matplotlib.pyplot as plt
from collections import Counter

#%%
# Generate Normal Distribution
np.random.seed(3101)  # For reproducibility
mu = 5  # Mean
sigma = np.sqrt(5)  # Standard deviation
n_samples = 50000

original_data = np.random.normal(mu, sigma, n_samples)

#%%
# Add noise
def add_noise(data, sigma_noise):
    noise = np.random.normal(0, sigma_noise, len(data))
    return data + noise

low_noise_data = add_noise(original_data, 0.5)
medium_noise_data = add_noise(original_data, 1.0)
high_noise_data = add_noise(original_data, 2.0)

#%%
# Combine data and assign labels
data = np.concatenate([original_data, low_noise_data, medium_noise_data, high_noise_data])
labels = np.array([0] * len(original_data) +
                  [1] * len(low_noise_data) +
                  [2] * len(medium_noise_data) +
                  [3] * len(high_noise_data))

#%%
# Function to calculate fractional part lengths
def fractional_part_lengths(data):
    lengths = []
    for value in data:
        # Extract fractional part as string
        fractional_str = str(value).split(".")[-1]
        lengths.append(len(fractional_str))
    return lengths

# Get the lengths for all values in the dataset
lengths = fractional_part_lengths(data)

# Count frequency of each length
length_counts = Counter(lengths)

# Prepare data for plotting
unique_lengths = sorted(length_counts.keys())
frequencies = [length_counts[length] for length in unique_lengths]

# Plot the distribution of lengths
plt.bar(unique_lengths, frequencies, color='skyblue', edgecolor='black')
plt.xlabel('Length of Fractional Part')
plt.ylabel('Frequency')
plt.title('Distribution of Fractional Part Lengths')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

#%%
# According to distribution of lengths write function to get ultimate and penultimate digits
def extract_features(data):
    int_part = np.floor(data).astype(int)  # Integer part
    fractional_part = data - int_part  # Fractional part

    # Correctly extract ultimate and penultimate digits
    ultimate_digit = ((fractional_part * 10**15) % 10).astype(int)
    penultimate_digit = ((fractional_part * 10**15) % 100 // 10).astype(int)

    return np.column_stack((int_part, ultimate_digit, penultimate_digit))

features = extract_features(data)
#%%
# Split data
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=3101)

#%%
# Train logistic regression
model = LogisticRegression(max_iter=1000, solver='lbfgs')
model.fit(X_train, y_train)

#%%
# Evaluate model
y_pred = model.predict(X_test)
print("Classification Report:")
print(classification_report(y_test, y_pred))

#%%
# Build ROC curves and calculate AUC for pairs of classes
pairs = [(3, 1), (1, 2), (2, 3)]
plt.figure(figsize=(10, 6))
for (class_a, class_b) in pairs:
    # Filter the data for the two classes
    idx = (y_test == class_a) | (y_test == class_b)
    X_test_pair = X_test[idx]
    y_test_pair = y_test[idx]
    y_pred_prob = model.predict_proba(X_test_pair)[:, class_a]  # Probabilities for class_a

    # Compute ROC curve
    fpr, tpr, _ = roc_curve(y_test_pair == class_a, y_pred_prob)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f"Class {class_a} vs Class {class_b} (AUC = {roc_auc:.2f})")

#%%
# Plot ROC curves
plt.title("ROC Curves")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend(loc="lower right")
plt.grid()
plt.show()

#%%
# Analysis
print("How does the level of noise affect classification?")
print("Higher noise levels lead to more distinct features, making classification easier.")