import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import gzip
import struct
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

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

train_labels_one_hot = -np.ones((train_labels.shape[0],10))
train_labels_one_hot[np.arange(train_labels_one_hot.shape[0]),train_labels]=1
test_labels_one_hot = -np.ones((test_labels.shape[0],10))
test_labels_one_hot[np.arange(test_labels_one_hot.shape[0]),test_labels]=1

iterations = 250

results = np.load('boosting_results.npz')
u_f = results['u_f']
u_t = results['u_t']
u_s = results['u_s']
u_e = results['u_e']
train_error = results['train_error']
test_error = results['test_error']
margin = results['margin']
w_max = results['w_max']
g_train = results['g_train']
g_test = results['g_test']

w_max = np.zeros((250,10,train_labels.shape[0]))

for it in range(iterations):
    for label in range(10):
        # compute the weights
        w  = np.exp(-margin[it,label,:])
        w_max[it,label,:] = w

err_rate = sum((np.argmax(g_test,axis=0)!= np.argmax(test_labels_one_hot,axis=1)))/test_labels.shape[0]
print("error rate: ",err_rate)

# For each binary classifier, plot train and test errors vs. iteration in a grid
fig, axs = plt.subplots(5, 2, figsize=(20, 20))
fig.suptitle('Train and Test Errors vs. Iteration for Each Digit Classifier', fontsize=20)
for label in range(10):
    ax = axs[label // 2, label % 2]
    ax.plot(train_error[:, label], label=f'Train Error (Digit {label})')
    ax.plot(test_error[:, label], label=f'Test Error (Digit {label})')
    ax.set_title(f'Digit {label}')
    ax.legend()
    print(test_error[:, label][-1])
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig('hw3figs/errors_vs_iteration.png')

# For each binary classifier, plot cumulative distribution function (cdf) of the margins in a grid
num_samples = train_labels.shape[0]
fig, axs = plt.subplots(5, 2, figsize=(20, 20))
fig.suptitle('Cumulative Distribution Function of the Margins for Each Digit Classifier', fontsize=20)
for label in range(10):
    ax = axs[label // 2, label % 2]
    for it in [5, 10, 50, 100, 250]:
        sorted_margins = np.sort(margin[it-1, label, :])
        cdf = np.arange(1, num_samples+1) / num_samples
        ax.plot(sorted_margins, cdf, label=f'Iteration {it}')
    ax.set_title(f'Digit {label}')
    ax.legend()
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig('hw3figs/cdf_of_margins.png')

# Visualize the weighting mechanism in a grid
fig = plt.figure(figsize=(20, 40))
outer_grid = GridSpec(10, 2, wspace=0.2, hspace=0.8)
fig.suptitle('Example of Largest Weight for Each Boosting Iteration', fontsize=20)
for label in range(10):
    inner_grid = outer_grid[label].subgridspec(1, 4, wspace=0.3, width_ratios=[5, 1, 1, 1])
    
    ax_main = fig.add_subplot(inner_grid[0])
    ax_main.plot(np.argmax(w_max[:, label, :], axis=1), label=f'Binary Classifier {label}')
    ax_main.set_title(f'Binary Classifier {label}')
    ax_main.legend()

    heaviest_indices = np.argsort(np.max(w_max[:, label, :], axis=0))[-3:]

    for i, idx in enumerate(heaviest_indices):
        ax_img = fig.add_subplot(inner_grid[i + 1])
        ax_img.imshow(train_images[idx].reshape(28, 28), cmap='gray')
        ax_img.set_title(f'Heaviest {i+1}')
        ax_img.axis('off')

plt.suptitle('Largest Weight Examples for Each Binary Classifier', fontsize=20)
plt.savefig('hw3figs/largest_weight_examples.png')

# Visualize what the weak learners do for each binary classifier in a grid
fig, axs = plt.subplots(5, 2, figsize=(20, 20))
fig.suptitle('Weak Learner Decisions for Each Binary Classifier', fontsize=20)
for label in range(10):
    a = 128 * np.ones((28, 28))
    for it in range(iterations):
        a[u_f[it, label] // 28, u_f[it, label] % 28] = 255 if u_s[it, label] == 1 else 0
    ax = axs[label // 2, label % 2]
    ax.imshow(a, cmap='gray')
    ax.set_title(f'Binary Classifier {label}')
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig('hw3figs/weak_learner_decisions.png')