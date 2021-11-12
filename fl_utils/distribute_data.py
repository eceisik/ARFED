import numpy as np
import pandas as pd
import torch
import cv2
import os

import os
from pathlib import Path
import requests
import pickle
import gzip
import math

import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt


# def load_mnist_data():
#     data_path = os.path.join("data", "mnist", "mnist.pkl.gz")
#     with gzip.open((data_path), "rb") as f:
#         ((x_train, y_train), (x_valid, y_valid), (x_test, y_test)) = pickle.load(f, encoding="latin-1")
#     return x_train, y_train, x_valid, y_valid, x_test, y_test


def load_mnist_data():
    DATA_PATH = Path("data")
    PATH = DATA_PATH / "mnist"

    PATH.mkdir(parents=True, exist_ok=True)

    URL = "https://github.com/pytorch/tutorials/raw/master/_static/"
    FILENAME = "mnist.pkl.gz"

    if not (PATH / FILENAME).exists():
        content = requests.get(URL + FILENAME).content
        (PATH / FILENAME).open("wb").write(content)

    with gzip.open((PATH / FILENAME).as_posix(), "rb") as f:
        ((x_train, y_train), (x_valid, y_valid), (x_test, y_test)) = pickle.load(f, encoding="latin-1")

    return x_train, y_train, x_valid, y_valid, x_test, y_test


def load_cifar_data():
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform)

    x_train = torch.zeros((50000, 3, 32, 32))
    y_train = torch.zeros(50000)
    ind_train = 0
    for data, output in trainset:
        x_train[ind_train, :, :, :] = data
        y_train[ind_train] = output
        ind_train = ind_train + 1

    x_test = torch.zeros((10000, 3, 32, 32))
    y_test = torch.zeros(10000)
    ind_test = 0
    for data, output in testset:
        x_test[ind_test, :, :, :] = data
        y_test[ind_test] = output
        ind_test = ind_test + 1

    y_train = y_train.type(torch.LongTensor)
    y_test = y_test.type(torch.LongTensor)

    return x_train, y_train, x_test, y_test



def show_grid_cifar(x_data,y_data, row,column):
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    fig, axes = plt.subplots(row,column,figsize=(8,8))
    for i in range(row):
        for j in range(column):
            num_index = np.random.randint(len(x_data))
            img=x_data[num_index,:,:,:]
            img = img / 2 + 0.5     # unnormalize
            npimg = img.numpy()
            npimg =np.transpose(npimg, (1, 2, 0))
            axes[i,j].imshow(npimg)

            axes[i,j].axis("off")
            axes[i,j].set_title(classes[int(y_data[num_index])])
    plt.show()

def load_fashion_mnist_data():
    transform = transforms.Compose([transforms.ToTensor()])

    trainset = torchvision.datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
    testset = torchvision.datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)

    x_train = trainset.data
    x_train = x_train / 255
    y_train = trainset.targets

    x_test = testset.data
    x_test = x_test / 255
    y_test = testset.targets

    return x_train, y_train, x_test, y_test


def show_grid_fashion_mnist(x_data, y_data, row, column):
    classes = ["T-shirt/Top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle Boot"]
    fig, axes = plt.subplots(row, column, figsize=(8, 8))
    for i in range(row):
        for j in range(column):
            num_index = np.random.randint(len(x_data))

            axes[i, j].imshow(x_data[num_index], cmap="gray")
            axes[i, j].axis("off")
            axes[i, j].set_title(classes[int(y_data[num_index])])
    plt.show()


def split_and_shuffle_labels(y_data, seed, amount):
    y_data = pd.DataFrame(y_data, columns=["labels"])
    y_data["i"] = np.arange(len(y_data))
    label_dict = dict()
    for i in range(10):
        var_name = "label" + str(i)
        label_info = y_data[y_data["labels"] == i]
        np.random.seed(seed)
        label_info = np.random.permutation(label_info)
        label_info = label_info[0:amount]
        label_info = pd.DataFrame(label_info, columns=["labels", "i"])
        label_dict.update({var_name: label_info})
    return label_dict



def get_info_for_distribute_non_iid_with_different_n_and_amount(number_of_samples, n, amount, seed, min_n_each_node=2):
    node_label_info = np.ones([number_of_samples, n]) * -1
    columns = []
    for j in range(n):
        columns.append("s" + str(j))
    node_label_info = pd.DataFrame(node_label_info, columns=columns, dtype=int)

    np.random.seed(seed)
    seeds = np.random.choice(number_of_samples * n * 5, size=number_of_samples, replace=False)
    for i in range(number_of_samples):
        np.random.seed(seeds[i])
        how_many_label_created = np.random.randint(
            n + 1 - min_n_each_node) + min_n_each_node  ## ensures at least one label is created by default
        which_labels = np.random.choice(10, size=how_many_label_created, replace=False)
        node_label_info.iloc[i, 0:len(which_labels)] = which_labels

    #################################
    #################################

    total_label_occurences = pd.DataFrame()
    for m in range(10):

        total_label_occurences.loc[0, m] = int(np.sum(node_label_info.values == m))
        if total_label_occurences.loc[0, m] == 0:
            total_label_occurences.loc[1, m] = 0
        else:
            total_label_occurences.loc[1, m] = int(amount / np.sum(node_label_info.values == m))
    total_label_occurences = total_label_occurences.astype('int32')

    ##################################
    ##################################

    amount_info_table = pd.DataFrame(np.zeros([number_of_samples, n]), dtype=int)
    for a in range(number_of_samples):
        for b in range(n):
            if node_label_info.iloc[a, b] == -1:
                amount_info_table.iloc[a, b] = 0
            else:
                amount_info_table.iloc[a, b] = total_label_occurences.iloc[1, node_label_info.iloc[a, b]]

    return node_label_info, total_label_occurences, amount_info_table


def distribute_mnist_data_to_participants(label_dict, amount, number_of_samples, n,
                                          x_data, y_data, x_name, y_name, node_label_info,
                                          amount_info_table, is_cnn=False):
    label_names = list(label_dict)
    label_dict_data = pd.DataFrame(columns=["labels", "i"])

    for a in label_names:
        data = pd.DataFrame.from_dict(label_dict[a])
        label_dict_data = pd.concat([label_dict_data, data], ignore_index=True)

    index_counter = pd.DataFrame(label_names, columns=["labels"])
    index_counter["start"] = np.ones(10, dtype=int) * np.arange(10) * amount
    index_counter["end"] = np.ones(10, dtype=int) * np.arange(10) * amount

    x_data_dict = dict()
    y_data_dict = dict()

    for i in range(number_of_samples):
        node_data_indices = pd.DataFrame()

        xname = x_name + str(i)
        yname = y_name + str(i)

        for j in range(n):
            label = node_label_info.iloc[i, j]
            if label != -1:
                label_amount = amount_info_table.iloc[i, j]
                index_counter.loc[label, "end"] = index_counter.loc[label, "end"] + label_amount
                node_data_indices = pd.concat([node_data_indices, label_dict_data.loc[
                                                                  index_counter.loc[label, "start"]:index_counter.loc[
                                                                                                        label, "end"] - 1,
                                                                  "i"]])
                index_counter.loc[label, "start"] = index_counter.loc[label, "end"]

        x_info = x_data[node_data_indices.iloc[:, 0].reset_index(drop=True), :]
        if is_cnn:
            reshape_size = int(np.sqrt(x_info.shape[1]))
            x_info = x_info.view(-1, 1, reshape_size, reshape_size)

        x_data_dict.update({xname: x_info})

        y_info = y_data[node_data_indices.iloc[:, 0].reset_index(drop=True)]
        y_data_dict.update({yname: y_info})

    return x_data_dict, y_data_dict


def distribute_fashion_data_to_participants(label_dict, amount, number_of_samples, n,
                                            x_data, y_data, x_name, y_name, node_label_info, amount_info_table):
    label_names = list(label_dict)
    label_dict_data = pd.DataFrame(columns=["labels", "i"])

    for a in label_names:
        data = pd.DataFrame.from_dict(label_dict[a])
        label_dict_data = pd.concat([label_dict_data, data], ignore_index=True)

    index_counter = pd.DataFrame(label_names, columns=["labels"])
    index_counter["start"] = np.ones(10, dtype=int) * np.arange(10) * amount
    index_counter["end"] = np.ones(10, dtype=int) * np.arange(10) * amount

    x_data_dict = dict()
    y_data_dict = dict()

    for i in range(number_of_samples):
        node_data_indices = pd.DataFrame()

        xname = x_name + str(i)
        yname = y_name + str(i)

        for j in range(n):
            label = node_label_info.iloc[i, j]
            if label != -1:
                label_amount = amount_info_table.iloc[i, j]
                index_counter.loc[label, "end"] = index_counter.loc[label, "end"] + label_amount
                node_data_indices = pd.concat([node_data_indices, label_dict_data.loc[
                                                                  index_counter.loc[label, "start"]:index_counter.loc[
                                                                                                        label, "end"] - 1,
                                                                  "i"]])
                #                 print(label, ", start:", index_counter.loc[label,"start"], ", end:", index_counter.loc[label,"end"] )
                index_counter.loc[label, "start"] = index_counter.loc[label, "end"]

        x_info = x_data[node_data_indices.iloc[:, 0].reset_index(drop=True), :]

        x_info = x_info.view(-1, 1, 28, 28)
        x_data_dict.update({xname: x_info})

        y_info = y_data[node_data_indices.iloc[:, 0].reset_index(drop=True)]
        y_data_dict.update({yname: y_info})

    return x_data_dict, y_data_dict


def distribute_cifar_data_to_participants(label_dict, amount, number_of_samples, n,
                                          x_data, y_data, x_name, y_name, node_label_info,
                                          amount_info_table):
    label_names = list(label_dict)
    label_dict_data = pd.DataFrame(columns=["labels", "i"])

    for a in label_names:
        data = pd.DataFrame.from_dict(label_dict[a])
        label_dict_data = pd.concat([label_dict_data, data], ignore_index=True)

    index_counter = pd.DataFrame(label_names, columns=["labels"])
    index_counter["start"] = np.ones(10, dtype=int) * np.arange(10) * amount
    index_counter["end"] = np.ones(10, dtype=int) * np.arange(10) * amount

    x_data_dict = dict()
    y_data_dict = dict()

    for i in range(number_of_samples):
        node_data_indices = pd.DataFrame()

        xname = x_name + str(i)
        yname = y_name + str(i)

        for j in range(n):
            label = node_label_info.iloc[i, j]
            if label != -1:
                label_amount = amount_info_table.iloc[i, j]
                index_counter.loc[label, "end"] = index_counter.loc[label, "end"] + label_amount
                node_data_indices = pd.concat([node_data_indices, label_dict_data.loc[
                                                                  index_counter.loc[label, "start"]:index_counter.loc[
                                                                                                        label, "end"] - 1,
                                                                  "i"]])

                index_counter.loc[label, "start"] = index_counter.loc[label, "end"]

        x_info = x_data[node_data_indices.iloc[:, 0].reset_index(drop=True), :]

        x_data_dict.update({xname: x_info})

        y_info = y_data[node_data_indices.iloc[:, 0].reset_index(drop=True)]
        y_data_dict.update({yname: y_info})

    return x_data_dict, y_data_dict


def create_just_data(x_data, y_data, x_just_name, y_just_name):
    x_just_dict = dict()
    y_just_dict = dict()

    for i in range(10):
        xname = x_just_name + str(i)
        x_info = x_data[y_data == i]
        x_just_dict.update({xname: x_info})

        yname = y_just_name + str(i)
        y_info = y_data[y_data == i]
        y_just_dict.update({yname: y_info})

    return x_just_dict, y_just_dict


def get_equal_size_test_data_from_each_label(x_test, y_test, min_amount=890):
    y_test_eq=pd.DataFrame(y_test, columns=["labels"])
    y_test_eq["ind"]=np.arange(len(y_test))
    hold=pd.DataFrame(columns=["labels", "ind"])
    for i in range(10):
        hold=pd.concat([hold,y_test_eq[y_test_eq["labels"]==i].iloc[0:min_amount,:] ])
    indices=np.array(hold["ind"], dtype=int)
    x_test=x_test[indices, :]
    y_test=y_test[indices]
    return x_test, y_test


def choose_nodes_randomly_to_convert_hostile(hostile_node_percentage, number_of_samples, hostility_seed=90):
    nodes_list=[]
    np.random.seed(hostility_seed)
    nodes=np.random.choice(number_of_samples, size=int(number_of_samples*hostile_node_percentage), replace=False)
    for node in nodes:
        name="y_train"+str(node)
        nodes_list.append(name)
    return nodes_list

def convert_nodes_to_hostile(y_dict, nodes_list,
                             converter_dict={0:9,1:7, 2:5,3:8, 4:6, 5:2, 6:4, 7:1, 8:3, 9:0}):
    for node in nodes_list:
        original_data=y_dict[node]
        converted_data=np.ones(y_dict[node].shape, dtype=int)*-1
        labels_in_node=np.unique(original_data)
        for label in labels_in_node:
            converted_data[original_data==label]=converter_dict[label]
        converted_data=(torch.tensor(converted_data)).type(torch.LongTensor)
        y_dict.update({node:converted_data})
    return y_dict



def create_different_converters_for_each_attacker(y_dict, nodes_list, converters_seed):
    converters = dict()
    np.random.seed(converters_seed)
    converter_seeds_array = np.random.choice(5000, size=len(nodes_list), replace=False)

    for i in range(len(nodes_list)):
        unique_labels = np.unique(y_dict[nodes_list[i]])
        np.random.seed([converter_seeds_array[i]])
        subseeds = np.random.choice(1000, len(unique_labels), replace=False)

        conv = dict()
        for j in range(len(unique_labels)):
            choose_from = np.delete(np.arange(10), unique_labels[j])
            np.random.seed(subseeds[j])
            chosen = np.random.choice(choose_from, replace=False)
            conv[unique_labels[j]] = chosen
        converters.update({nodes_list[i]: conv})
    return converters

def convert_nodes_to_hostile_with_different_converters(y_dict, nodes_list, converters_seed=61):
    converters= create_different_converters_for_each_attacker(y_dict, nodes_list, converters_seed)
    y_dict_converted = y_dict.copy()
    for node in nodes_list:
        original_data=y_dict[node]
        converted_data=np.ones(y_dict[node].shape, dtype=int)*-1
        labels_in_node=np.unique(original_data)
        for label in labels_in_node:
            converted_data[original_data==label]=converters[node][label]
        converted_data=(torch.tensor(converted_data)).type(torch.LongTensor)
        y_dict_converted.update({node:converted_data})
    return y_dict_converted


def get_byzantine_node_list(hostile_node_percentage, number_of_samples, hostility_seed=90):

    np.random.seed(hostility_seed)
    nodes=np.random.choice(number_of_samples, size=int(number_of_samples*hostile_node_percentage), replace=False)
    return nodes