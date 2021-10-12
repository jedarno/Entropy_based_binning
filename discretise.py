import numpy as np

def entropy(X):
  labels, counts = np.unique(X[:,1], return_counts = 1)
  num_of_labels = labels.size
  entropy = 0

  for i in range(num_of_labels):
    p = counts[i]/X.shape[0]
    #print(p)
    entropy -= p * np.log2(p)
  return entropy

def information_gain(X, bin1, bin2):

  entropy_X = entropy(X)

  ratio_X_bin1 = bin1.shape[0]/X.shape[0]
  ratio_X_bin2 = bin2.shape[0]/X.shape[0]

  entropy_bin1 = entropy(bin1)
  entropy_bin2 = entropy(bin2)


  info_gained = entropy_X - ((ratio_X_bin1 * entropy(bin1)) + (ratio_X_bin2 * entropy(bin2)))

  return info_gained

def discretise(X, min_size = 2, entropy_threshold = 0.1):
  entropy_X = entropy(X)
  best_i = -1
  best_gain = 0

  if X.shape[0] <= 2 or entropy_X < entropy_threshold:
    return X

  for i in range(1, X.shape[0]):
    bin1 = X[:i, :]
    bin2 = X[i:, :]
    info_gain = information_gain(X, bin1, bin2)

    if info_gain > best_gain:
      best_gain = info_gain
      best_i = i

  return discretise(X[:best_i, :], min_size, entropy_threshold), discretise(X[best_i:, :], min_size, entropy_threshold)
