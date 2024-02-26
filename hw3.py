import numpy as np
import gzip
import struct
import matplotlib.pyplot as plt

############# translate matlab code to python code ###################
def readMNIST(image_file, label_file, readDigits, offset):
    with gzip.open(image_file, 'rb') as f:
        magic, num_images, rows, cols = struct.unpack(">IIII", f.read(16))
        if magic != 2051:
            raise ValueError("Invalid image file header")
        if num_images < readDigits + offset:
            raise ValueError("Trying to read too many digits")

        f.seek(rows * cols * offset, 1)  # Seek to the start of image data with offset

        # Read image data
        imgs = np.frombuffer(f.read(readDigits * rows * cols), dtype=np.uint8)
        imgs = imgs.reshape(readDigits, rows * cols)  # Reshape to (num_images, 784)

    with gzip.open(label_file, 'rb') as f:
        magic, num_labels = struct.unpack(">II", f.read(8))
        if magic != 2049:
            raise ValueError("Invalid label file header")
        if num_labels < readDigits + offset:
            raise ValueError("Trying to read too many digits")

        f.seek(offset, 1)  # Seek to the start of label data with offset

        # Read label data
        labels = np.frombuffer(f.read(readDigits), dtype=np.uint8)

    # Normalize pixel values
    imgs = normalizePixValue(imgs)

    return imgs, labels

def normalizePixValue(digits):
    return digits.astype(np.float32) / 255.0


train_images_file = 'quiz2data/training set/train-images-idx3-ubyte.gz'
train_labels_file = 'quiz2data/training set/train-labels-idx1-ubyte.gz'
test_images_file = 'quiz2data/test set/t10k-images-idx3-ubyte.gz'
test_labels_file = 'quiz2data/test set/t10k-labels-idx1-ubyte.gz'

train_images, train_labels = readMNIST(train_images_file, train_labels_file, 20000, 0)
test_images, test_labels = readMNIST(test_images_file, test_labels_file, 10000, 0)
#%%
#ones_column = -np.ones((train_images.shape[0], 1))
#train_images = np.hstack((train_images, ones_column))
train_labels_one_hot = -np.ones((train_labels.shape[0],10))
train_labels_one_hot[np.arange(train_labels_one_hot.shape[0]),train_labels]=1

#ones_column = -np.ones((test_images.shape[0], 1))
#test_images = np.hstack((test_images, ones_column))
test_labels_one_hot = -np.ones((test_labels.shape[0],10))
test_labels_one_hot[np.arange(test_labels_one_hot.shape[0]),test_labels]=1
#%%
itations = 10
g_train = np.random.randint(0, 2, (10,train_images.shape[0]))
g_test = np.random.randint(0, 2, (10, test_images.shape[0]))
w = np.zeros((train_images.shape[0],1))
w_max = np.zeros((itations,1))
margin = np.zeros((itations,10,train_images.shape[0]))
t = np.linspace(0,1,51)
u_t = np.zeros((itations,10))
u_f = np.zeros((itations,10)).astype(int)
u_s = np.zeros((itations,10))
u_e = np.zeros((itations,10))
train_error = np.zeros((itations,10))
test_error = np.zeros((itations,10))
#%%
def decision_stump(data,threshold,feature,sign):

    return sign*(2*(data[:,feature]>threshold)-1)

def train_decision_stump(data,labels,w):
    best_acc = - np.inf
    for feature in range(data.shape[1]):
        for threshold in np.linspace(0,1,51):
            for sign in [-1,1]:
                weighted_acc = np.sum(w*(decision_stump(data,threshold,feature,sign)*labels))
                if weighted_acc > best_acc:
                    best_acc = weighted_acc
                    best_feature = feature
                    best_threshold = threshold
                    best_sign = sign
    return best_feature, best_threshold, best_sign, best_acc

def train_decision_stump_(data, labels, w):
    num_samples, num_features = data.shape
    thresholds = np.linspace(0, 1, 51)

    best_acc = -np.inf
    best_feature = None
    best_threshold = None
    best_sign = None

    # 对每个特征进行并行计算
    for feature in range(num_features):
        feature_data = data[:, feature]

        for threshold in thresholds:
            for sign in [-1, 1]:
                # 使用广播计算当前特征、当前阈值和符号的决策结果
                decision_results = sign * np.sign(feature_data - threshold)

                # 计算加权准确率
                weighted_acc = np.sum(w * (decision_results == labels))

                if weighted_acc > best_acc:
                    best_acc = weighted_acc
                    best_feature = feature
                    best_threshold = threshold
                    best_sign = sign

    return best_feature, best_threshold, best_sign, best_acc
#%%
from tqdm import tqdm

for it in tqdm(range(itations)):
    for label in tqdm(range(10)):
        # compute the weights
        margin[it,label,:] = g_train[label,:]*train_labels_one_hot[:,label]
        w  = np.exp(-margin[it,label,:])
        w_max[it] = np.max(w)

        # compute the negative gradient
        u_f[it,label], u_t[it,label], u_s[it,label], u_e[it,label] = train_decision_stump(train_images,train_labels_one_hot[:,label],w)
        alpha_t = decision_stump(train_images,u_t[it,label],u_f[it,label],u_s[it,label])

        # compute the setp-size
        epsilon = np.sum(w*(alpha_t!=train_labels_one_hot[:,label]))/np.sum(w)
        wt = np.log((1-epsilon)/epsilon)

        # update the learned function
        g_train[label,:] = g_train[label,:] + wt*alpha_t
        g_test[label,:] = g_test[label,:] + wt*decision_stump(test_images,u_t[it,label],u_f[it,label],u_s[it,label])

        # compute the train and test error
        train_error[it,label] = np.sum(np.sign(g_train[label,:])!=train_labels_one_hot[:,label])/train_images.shape[0]
        test_error[it,label] = np.sum(np.sign(g_test[label,:])!=test_labels_one_hot[:,label])/test_images.shape[0]

# save the results
np.savez('boosting_results.npz', u_f=u_f, u_t=u_t, u_s=u_s, u_e=u_e, train_error=train_error, test_error=test_error, margin=margin, w_max=w_max, g_train=g_train, g_test=g_test)
#%% md

#%%
# For each binary classifier, plot train and test errors vs. iteration.
plt.figure(figsize=(10, 10))
for label in range(10):
    plt.plot(train_error[:, label], label=f'Train error (digit {label})')
    plt.plot(test_error[:, label], label=f'Test error (digit {label})')
#%%
# For each binary classifier, make a plot of cumulative distribution function (cdf) of the margins γi of all training examples after {5, 10, 50, 100, 250} iterations (the cdf is the function F(a) = P(γ ≤ a), and you can do this by computing an histogram and then a cumulative sum of histogram bins.)
plt.figure(figsize=(10, 10))
for label in range(10):
    for it in [5, 10, 50, 100, 250]:
        plt.hist(margin[it, label, :], bins=50, density=True, histtype='step', cumulative=True, label=f'Iteration {it}')
    plt.title(f'Cumulative distribution function of the margins (digit {label})')
    plt.legend()
    plt.show()
#%%
# We now visualize the weighting mechanism. For each of the 10 binary classifiers, do the following.  Make a plot of the index of the example of largest weight for each boosting iteration.  Plot the three “heaviest” examples. These are the 3 examples that were most frequently selected, across boosting iterations, as the example of largest weight.
plt.figure(figsize=(10, 10))
for label in range(10):
    plt.plot(np.argmax(w, axis=1), label=f'Binary classifier {label}')
plt.title('Index of the example of largest weight for each boosting iteration')
plt.legend()
plt.show()
# plot the 3 “heaviest” examples
plt.figure(figsize=(10, 10))
for label in range(10):
    heaviest_examples = np.argsort(np.sum(w, axis=0))[-3:]
    for example in heaviest_examples:
        plt.imshow(train_images[example].reshape(28, 28), cmap='gray')
        plt.title(f'Binary classifier {label}')
        plt.show()
#%%
#We now visualize what the weak learners do. Consider the weak learners αk, k ∈ {1, . . . , K}, chosen by each iteration of AdaBoost. Let i(k) be the index of the feature xi(k) selected at time k. Note that i(k) corresponds to an image (pixel) location. Create a 28 × 28 array a filled with the value 128. Then, for αk, k ∈ {1, . . . , K}, do the following.  If the weak learner selected at iteration k is a regular weak learner (outputs 1 for xi(k) greater than its threshold), store the value 255 on location i(k) of array a.  If the weak learner selected at iteration k is a twin weak learner (outputs −1 for xi(k) greater than its threshold), store the value 0 on location i(k) of array a. Create the array a for each of the 10 binary classifiers and make a plot of the 10 arrays. Comment on what the classifiers are doing to reach their classification decision.
a = 128 * np.ones((28, 28))
for label in range(10):
    for it in range(itations):
        # map feature index to the position in the image
        a[u_f[it, label] // 28, u_f[it, label] % 28] = 255 if u_s[it, label] == 1 else 0
    plt.imshow(a, cmap='gray')
    plt.title(f'Binary classifier {label}')
    plt.show()
