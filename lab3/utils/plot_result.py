import matplotlib.pyplot as plt
import numpy as np
import os

def plot_result(epochs, acc_train_dict, acc_test_dict, title, img_name):
    # fig, ax = plt.subplots()  # Create a figure and an axes.
    
    x = np.arange(1, epochs+1)
    plt.plot(x, acc_train_dict["ReLU"], label='relu_train')  # Plot some data on the axes.
    plt.plot(x, acc_test_dict["ReLU"], label='relu_test')  # Plot some data on the axes.
    plt.plot(x, acc_train_dict["LeakyReLU"], label='leaky_relu_train')  # Plot more data on the axes...
    plt.plot(x, acc_test_dict["LeakyReLU"], label='leaky_relu_test')  # ... and some more.
    plt.plot(x, acc_train_dict["ELU"], label='elu_train')  # Plot more data on the axes...
    plt.plot(x, acc_test_dict["ELU"], label='elu_test')  # ... and some more.
    plt.ylabel('Accuracy(%)')  # Add an x-label to the axes.
    plt.xlabel('Epoch')  # Add a y-label to the axes.
    title = "Activation function comparison (" + title + ")"
    plt.title(title)  # Add a title to the axes.
    plt.legend()  # Add a legend.
    folder = img_name.split('/')[0]
    if not os.path.exists(folder):
        os.makedirs(folder)
    plt.savefig(img_name)