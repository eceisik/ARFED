import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
from torchvision import models
from torchvision import transforms
from fl_utils import construct_models as cm
from statistics import NormalDist
from scipy import stats


def train(model, train_loader, criterion, optimizer, device):
    model.train()
    train_loss = 0.0
    correct = 0

    for data, target in train_loader:
        data, target = data.to(device), target.to(device)
        output = model(data)
        loss = criterion(output, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        prediction = output.argmax(dim=1, keepdim=True)
        correct += prediction.eq(target.view_as(prediction)).sum().item()

    return train_loss / len(train_loader), correct / len(train_loader.dataset)

def train_with_clipping(model, train_loader, criterion, optimizer, device, clipping=True, clipping_threshold=10):
    model.train()
    train_loss = 0.0
    correct = 0

    for data, target in train_loader:
        data, target = data.to(device), target.to(device)
        output = model(data)
        loss = criterion(output, target)
        optimizer.zero_grad()
        loss.backward()
        if clipping:
            torch.nn.utils.clip_grad_value_(model.parameters(), clipping_threshold)
        optimizer.step()

        train_loss += loss.item()
        prediction = output.argmax(dim=1, keepdim=True)
        correct += prediction.eq(target.view_as(prediction)).sum().item()

    return train_loss / len(train_loader), correct / len(train_loader.dataset)


def train_with_augmentation(model, train_loader, criterion, optimizer, device, clipping, clipping_threshold=10, use_augmentation=False, augment=None ):
    model.train()
    train_loss = 0.0
    correct = 0

    for data, target in train_loader:
        data, target = data.to(device), target.to(device)

        if use_augmentation:
            data = augment(data)

        output = model(data)
        loss = criterion(output, target)
        optimizer.zero_grad()
        loss.backward()
        if clipping:
            torch.nn.utils.clip_grad_value_(model.parameters(), clipping_threshold)

        optimizer.step()

        train_loss += loss.item()
        prediction = output.argmax(dim=1, keepdim=True)
        correct += prediction.eq(target.view_as(prediction)).sum().item()

    return train_loss / len(train_loader), correct / len(train_loader.dataset)


def validation(model, test_loader, criterion, device):
    model.eval()
    test_loss = 0.0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)

            test_loss += criterion(output, target).item()
            prediction = output.argmax(dim=1, keepdim=True)
            correct += prediction.eq(target.view_as(prediction)).sum().item()

    test_loss /= len(test_loader)
    correct /= len(test_loader.dataset)

    return (test_loss, correct)


def get_model_names(model_dict):
    name_of_models = list(model_dict.keys())
    return name_of_models


def get_optimizer_names(optimizer_dict):
    name_of_optimizers = list(optimizer_dict.keys())
    return name_of_optimizers


def get_criterion_names(criterion_dict):
    name_of_criterions = list(criterion_dict.keys())
    return name_of_criterions


def get_x_train_sets_names(x_train_dict):
    name_of_x_train_sets = list(x_train_dict.keys())
    return name_of_x_train_sets


def get_y_train_sets_names(y_train_dict):
    name_of_y_train_sets = list(y_train_dict.keys())
    return name_of_y_train_sets


def get_x_valid_sets_names(x_valid_dict):
    name_of_x_valid_sets = list(x_valid_dict.keys())
    return name_of_x_valid_sets


def get_y_valid_sets_names(y_valid_dict):
    name_of_y_valid_sets = list(y_valid_dict.keys())
    return name_of_y_valid_sets


def get_x_test_sets_names(x_test_dict):
    name_of_x_test_sets = list(x_test_dict.keys())
    return name_of_x_test_sets


def get_y_test_sets_names(y_test_dict):
    name_of_y_test_sets = list(y_test_dict.keys())
    return name_of_y_test_sets


def create_model_optimizer_criterion_dict_for_mnist(number_of_samples, learning_rate, momentum, device, is_cnn=False, weight_decay=0):
    model_dict = dict()
    optimizer_dict = dict()
    criterion_dict = dict()

    for i in range(number_of_samples):
        model_name = "model" + str(i)
        if is_cnn:
            model_info = cm.Netcnn()
        else:
            model_info = cm.Net2nn()
        model_info = model_info.to(device)
        model_dict.update({model_name: model_info})

        optimizer_name = "optimizer" + str(i)
        optimizer_info = torch.optim.SGD(model_info.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
        optimizer_dict.update({optimizer_name: optimizer_info})

        criterion_name = "criterion" + str(i)
        criterion_info = nn.CrossEntropyLoss()
        criterion_dict.update({criterion_name: criterion_info})

    return model_dict, optimizer_dict, criterion_dict


def create_model_optimizer_criterion_dict_for_cifar_net(number_of_samples, learning_rate, momentum, device,
                                                        weight_decay=0):
    model_dict = dict()
    optimizer_dict = dict()
    criterion_dict = dict()

    for i in range(number_of_samples):
        model_name = "model" + str(i)

        model_info = cm.Netcnn_cifar()

        model_info = model_info.to(device)
        model_dict.update({model_name: model_info})

        optimizer_name = "optimizer" + str(i)
        optimizer_info = torch.optim.SGD(model_info.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
        optimizer_dict.update({optimizer_name: optimizer_info})

        criterion_name = "criterion" + str(i)
        criterion_info = nn.CrossEntropyLoss()
        criterion_dict.update({criterion_name: criterion_info})

    return model_dict, optimizer_dict, criterion_dict

def create_model_optimizer_criterion_dict_for_cifar_cnn(number_of_samples, learning_rate, momentum, device,
                                                        weight_decay=0):
    model_dict = dict()
    optimizer_dict = dict()
    criterion_dict = dict()

    for i in range(number_of_samples):
        model_name = "model" + str(i)

        model_info = cm.Cifar10CNN()

        model_info = model_info.to(device)
        model_dict.update({model_name: model_info})

        optimizer_name = "optimizer" + str(i)
        optimizer_info = torch.optim.SGD(model_info.parameters(), lr=learning_rate, momentum=momentum,
                                         weight_decay=weight_decay)
        optimizer_dict.update({optimizer_name: optimizer_info})

        criterion_name = "criterion" + str(i)
        criterion_info = nn.CrossEntropyLoss()
        criterion_dict.update({criterion_name: criterion_info})

    return model_dict, optimizer_dict, criterion_dict

def create_model_optimizer_criterion_dict_for_fashion_mnist(number_of_samples, learning_rate, momentum, device, weight_decay=0):
    model_dict = dict()
    optimizer_dict = dict()
    criterion_dict = dict()

    for i in range(number_of_samples):
        model_name = "model" + str(i)

        model_info = cm.Net_fashion()
        model_info = model_info.to(device)
        model_dict.update({model_name: model_info})

        optimizer_name = "optimizer" + str(i)
        optimizer_info = torch.optim.SGD(model_info.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
        optimizer_dict.update({optimizer_name: optimizer_info})

        criterion_name = "criterion" + str(i)
        criterion_info = nn.CrossEntropyLoss()
        criterion_dict.update({criterion_name: criterion_info})

    return model_dict, optimizer_dict, criterion_dict


def create_model_optimizer_criterion_dict_for_cifar_resnet(number_of_samples, learning_rate, momentum, device,
                                                           weight_decay=0):
    model_dict = dict()
    optimizer_dict = dict()
    criterion_dict = dict()

    for i in range(number_of_samples):
        model_name = "model" + str(i)

        model_info = models.resnet18(num_classes=10)

        model_info = model_info.to(device)
        model_dict.update({model_name: model_info})

        optimizer_name = "optimizer" + str(i)
        optimizer_info = torch.optim.SGD(model_info.parameters(), lr=learning_rate, momentum=momentum,
                                         weight_decay=weight_decay)
        optimizer_dict.update({optimizer_name: optimizer_info})

        criterion_name = "criterion" + str(i)
        criterion_info = nn.CrossEntropyLoss()
        criterion_dict.update({criterion_name: criterion_info})

    return model_dict, optimizer_dict, criterion_dict


def send_main_model_to_nodes_and_update_model_dict(main_model, model_dict, number_of_samples):
    name_of_models = list(model_dict.keys())
    main_model_param_data_list = list(main_model.parameters())
    with torch.no_grad():
        for i in range(number_of_samples):
            sample_param_data_list = list(model_dict[name_of_models[i]].parameters())
            for j in range(len(main_model_param_data_list)):
                sample_param_data_list[j].data = main_model_param_data_list[j].data.clone()
    return model_dict


def compare_local_and_merged_model_performance(number_of_samples, x_test_dict, y_test_dict,
                                               batch_size, model_dict, criterion_dict, main_model,
                                               main_criterion, device):
    accuracy_table = pd.DataFrame(data=np.zeros((number_of_samples, 3)),
                                  columns=["sample", "local_ind_model", "merged_main_model"])

    name_of_x_test_sets = list(x_test_dict.keys())
    name_of_y_test_sets = list(y_test_dict.keys())

    name_of_models = list(model_dict.keys())
    name_of_criterions = list(criterion_dict.keys())

    for i in range(number_of_samples):
        test_ds = TensorDataset(x_test_dict[name_of_x_test_sets[i]], y_test_dict[name_of_y_test_sets[i]])
        test_dl = DataLoader(test_ds, batch_size=batch_size * 2)

        model = model_dict[name_of_models[i]]
        criterion = criterion_dict[name_of_criterions[i]]

        individual_loss, individual_accuracy = validation(model, test_dl, criterion, device)
        main_loss, main_accuracy = validation(main_model, test_dl, main_criterion, device)

        accuracy_table.loc[i, "sample"] = "sample " + str(i)
        accuracy_table.loc[i, "local_ind_model"] = individual_accuracy
        accuracy_table.loc[i, "merged_main_model"] = main_accuracy

    return accuracy_table


def start_train_end_node_process_without_print(number_of_samples, x_train_dict, y_train_dict, x_test_dict, y_test_dict,
                                               batch_size, model_dict, criterion_dict, optimizer_dict, numEpoch,
                                               device):
    name_of_x_train_sets = get_x_train_sets_names(x_train_dict)
    name_of_y_train_sets = get_y_train_sets_names(y_train_dict)
    name_of_x_test_sets = get_x_test_sets_names(x_test_dict)
    name_of_y_test_sets = get_y_test_sets_names(y_test_dict)
    name_of_models = get_model_names(model_dict)
    name_of_criterions = get_criterion_names(criterion_dict)
    name_of_optimizers = get_optimizer_names(optimizer_dict)

    for i in range(number_of_samples):

        train_ds = TensorDataset(x_train_dict[name_of_x_train_sets[i]], y_train_dict[name_of_y_train_sets[i]])
        train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

        test_ds = TensorDataset(x_test_dict[name_of_x_test_sets[i]], y_test_dict[name_of_y_test_sets[i]])
        test_dl = DataLoader(test_ds, batch_size=batch_size * 2)

        model = model_dict[name_of_models[i]]

        criterion = criterion_dict[name_of_criterions[i]]
        optimizer = optimizer_dict[name_of_optimizers[i]]

        for epoch in range(numEpoch):
            train_loss, train_accuracy = train(model, train_dl, criterion, optimizer, device)
            test_loss, test_accuracy = validation(model, test_dl, criterion, device)

def start_train_end_node_process_with_cliiping(number_of_samples, x_train_dict, y_train_dict, x_test_dict, y_test_dict,
                                               batch_size, model_dict, criterion_dict, optimizer_dict, numEpoch,
                                               device, clipping=True, clipping_threshold=10):
    name_of_x_train_sets = get_x_train_sets_names(x_train_dict)
    name_of_y_train_sets = get_y_train_sets_names(y_train_dict)
    name_of_x_test_sets = get_x_test_sets_names(x_test_dict)
    name_of_y_test_sets = get_y_test_sets_names(y_test_dict)
    name_of_models = get_model_names(model_dict)
    name_of_criterions = get_criterion_names(criterion_dict)
    name_of_optimizers = get_optimizer_names(optimizer_dict)

    for i in range(number_of_samples):

        train_ds = TensorDataset(x_train_dict[name_of_x_train_sets[i]], y_train_dict[name_of_y_train_sets[i]])
        train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

        test_ds = TensorDataset(x_test_dict[name_of_x_test_sets[i]], y_test_dict[name_of_y_test_sets[i]])
        test_dl = DataLoader(test_ds, batch_size=batch_size * 2)

        model = model_dict[name_of_models[i]]

        criterion = criterion_dict[name_of_criterions[i]]
        optimizer = optimizer_dict[name_of_optimizers[i]]

        for epoch in range(numEpoch):
            train_loss, train_accuracy = train_with_clipping(model, train_dl, criterion, optimizer, device, clipping, clipping_threshold)

            test_loss, test_accuracy = validation(model, test_dl, criterion, device)



def start_train_end_node_process_cifar(number_of_samples, x_train_dict, y_train_dict, x_test_dict, y_test_dict,
                                       batch_size, model_dict, criterion_dict, optimizer_dict, numEpoch,
                                       device,clipping=False, clipping_threshold =10):
    name_of_x_train_sets = get_x_train_sets_names(x_train_dict)
    name_of_y_train_sets = get_y_train_sets_names(y_train_dict)
    name_of_x_test_sets = get_x_test_sets_names(x_test_dict)
    name_of_y_test_sets = get_y_test_sets_names(y_test_dict)
    name_of_models = get_model_names(model_dict)
    name_of_criterions = get_criterion_names(criterion_dict)
    name_of_optimizers = get_optimizer_names(optimizer_dict)

    transform_augment = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop((32, 32), padding=4)])

    for i in range(number_of_samples):

        train_ds = TensorDataset(x_train_dict[name_of_x_train_sets[i]], y_train_dict[name_of_y_train_sets[i]])
        train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

        test_ds = TensorDataset(x_test_dict[name_of_x_test_sets[i]], y_test_dict[name_of_y_test_sets[i]])
        test_dl = DataLoader(test_ds, batch_size=batch_size * 2)

        model = model_dict[name_of_models[i]]
        criterion = criterion_dict[name_of_criterions[i]]
        optimizer = optimizer_dict[name_of_optimizers[i]]

        for epoch in range(numEpoch):
            train_loss, train_accuracy = train_with_augmentation(model, train_dl, criterion, optimizer, device,
                                                                 clipping=clipping,
                                                                 clipping_threshold=clipping_threshold,
                                                                 use_augmentation=True, augment=transform_augment)

            test_loss, test_accuracy = validation(model, test_dl, criterion, device)


##########################################

def start_train_end_node_process_byzantine_for_cifar_with_augmentation(number_of_samples, x_train_dict, y_train_dict, x_test_dict, y_test_dict,
                                               batch_size, model_dict, criterion_dict, optimizer_dict, numEpoch, byzantine_node_list,
                                            byzantine_mean, byzantine_std, device, clipping=False, clipping_threshold =10, iteration_byzantine_seed=None ):

    name_of_x_train_sets = get_x_train_sets_names(x_train_dict)
    name_of_y_train_sets = get_y_train_sets_names(y_train_dict)
    name_of_x_test_sets = get_x_test_sets_names(x_test_dict)
    name_of_y_test_sets = get_y_test_sets_names(y_test_dict)
    name_of_models = get_model_names(model_dict)
    name_of_criterions = get_criterion_names(criterion_dict)
    name_of_optimizers = get_optimizer_names(optimizer_dict)

    transform_augment = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop((32, 32), padding=4)])



    trusted_nodes=  np.array(list(set(np.arange(number_of_samples)) - set(byzantine_node_list)), dtype=int)

    ## STANDARD LOCAL MODEL TRAİNİNG PROCESS FOR TRUSTED NODES
    for i in trusted_nodes:

        train_ds = TensorDataset(x_train_dict[name_of_x_train_sets[i]], y_train_dict[name_of_y_train_sets[i]])
        train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

        test_ds = TensorDataset(x_test_dict[name_of_x_test_sets[i]], y_test_dict[name_of_y_test_sets[i]])
        test_dl = DataLoader(test_ds, batch_size=batch_size * 2)

        model = model_dict[name_of_models[i]]
        criterion = criterion_dict[name_of_criterions[i]]
        optimizer = optimizer_dict[name_of_optimizers[i]]

        for epoch in range(numEpoch):
            train_loss, train_accuracy = train_with_augmentation(model, train_dl, criterion, optimizer, device, clipping=clipping, clipping_threshold=clipping_threshold,
                                                                 use_augmentation=True, augment=transform_augment)
            test_loss, test_accuracy = validation(model, test_dl, criterion, device)

    with torch.no_grad():

        for j in byzantine_node_list:

            hostile_node_param_data_list = list(model_dict[name_of_models[j]].parameters())

            for k in range(len(hostile_node_param_data_list)):
                np.random.seed(iteration_byzantine_seed)
                hostile_node_param_data_list[k].data = torch.tensor(np.random.normal(byzantine_mean,byzantine_std, hostile_node_param_data_list[k].data.shape ), dtype=torch.float32, device=device)

            model_dict[name_of_models[j]].float()


###############################################

def start_train_end_node_process_byzantine(number_of_samples, x_train_dict, y_train_dict, x_test_dict, y_test_dict,
                                               batch_size, model_dict, criterion_dict, optimizer_dict,
                                               numEpoch, byzantine_node_list, byzantine_mean, byzantine_std, device, iteration_byzantine_seed=None ):
    name_of_x_train_sets = get_x_train_sets_names(x_train_dict)
    name_of_y_train_sets = get_y_train_sets_names(y_train_dict)
    name_of_x_test_sets = get_x_test_sets_names(x_test_dict)
    name_of_y_test_sets = get_y_test_sets_names(y_test_dict)
    name_of_models = get_model_names(model_dict)
    name_of_criterions = get_criterion_names(criterion_dict)
    name_of_optimizers = get_optimizer_names(optimizer_dict)



    trusted_nodes=  np.array(list(set(np.arange(number_of_samples)) - set(byzantine_node_list)), dtype=int)

    ## STANDARD LOCAL MODEL TRAİNİNG PROCESS FOR TRUSTED NODES
    for i in trusted_nodes:

        train_ds = TensorDataset(x_train_dict[name_of_x_train_sets[i]], y_train_dict[name_of_y_train_sets[i]])
        train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

        test_ds = TensorDataset(x_test_dict[name_of_x_test_sets[i]], y_test_dict[name_of_y_test_sets[i]])
        test_dl = DataLoader(test_ds, batch_size=batch_size * 2)

        model = model_dict[name_of_models[i]]
        criterion = criterion_dict[name_of_criterions[i]]
        optimizer = optimizer_dict[name_of_optimizers[i]]

        for epoch in range(numEpoch):
            train_loss, train_accuracy = train(model, train_dl, criterion, optimizer, device)
            test_loss, test_accuracy = validation(model, test_dl, criterion, device)


    with torch.no_grad():
        for j in byzantine_node_list:
            hostile_node_param_data_list = list(model_dict[name_of_models[j]].parameters())

            for k in range(len(hostile_node_param_data_list)):
                np.random.seed(iteration_byzantine_seed)
                hostile_node_param_data_list[k].data = torch.tensor(np.random.normal(byzantine_mean,byzantine_std, hostile_node_param_data_list[k].data.shape ), dtype=torch.float32, device=device)

            model_dict[name_of_models[j]].float()


################################################

def calculate_euclidean_distances(main_model, model_dict):
    calculated_parameter_names = []

    for parameters in main_model.named_parameters():  ## bias dataları için distance hesaplamıyorum
        if "bias" not in parameters[0]:
            calculated_parameter_names.append(parameters[0])

    columns = ["model"] + calculated_parameter_names
    distances = pd.DataFrame(columns=columns)
    model_names = list(model_dict.keys())

    main_model_weight_dict = {}
    for parameter in main_model.named_parameters():
        name = parameter[0]
        weight_info = parameter[1]
        main_model_weight_dict.update({name: weight_info})

    with torch.no_grad():
        for i in range(len(model_names)):
            distances.loc[i, "model"] = model_names[i]
            sample_node_parameter_list = list(model_dict[model_names[i]].named_parameters())
            for j in sample_node_parameter_list:
                if j[0] in calculated_parameter_names:
                    distances.loc[i, j[0]] = round(
                        np.linalg.norm(main_model_weight_dict[j[0]].cpu().data - j[1].cpu().data), 4)

    return distances


def calculate_lower_and_upper_limit(data, factor):
    quantiles = data.quantile(q=[0.25, 0.50, 0.75]).values
    q1 = quantiles[0]
    q2 = quantiles[1]
    q3 = quantiles[2]
    iqr = q3 - q1
    lower_limit = q1 - factor * iqr
    upper_limit = q3 + factor * iqr
    return lower_limit, upper_limit


def get_outlier_situation_and_thresholds_for_layers(distances, factor=1.5):
    layers = list(distances.columns)
    layers.remove("model")
    threshold_columns = []
    for layer in layers:
        threshold_columns.append((layer + "_lower"))
        threshold_columns.append((layer + "_upper"))
    thresholds = pd.DataFrame(columns=threshold_columns)

    include_calculation_result = True
    for layer in layers:
        data = distances[layer]
        lower, upper = calculate_lower_and_upper_limit(data, factor)
        lower_name = layer + "_lower"
        upper_name = layer + "_upper"
        thresholds.loc[0, lower_name] = lower
        thresholds.loc[0, upper_name] = upper
        name = layer + "_is_in_ci"

        distances[name] = (distances[layer] > lower) & (distances[layer] < upper)
        include_calculation_result = include_calculation_result & distances[name]

    distances["include_calculation"] = include_calculation_result
    return distances, thresholds


def compare_individual_models_on_only_one_label(model_dict, criterion_dict, x_just_dict, y_just_dict, batch_size,
                                                device):
    columns = ["model_name"]
    label_names = []
    for l in range(10):
        label_names.append("label" + str(l))
        columns.append("label" + str(l))

    accuracy_rec = pd.DataFrame(data=np.zeros([10, 11]), columns=columns)

    # x_just_dict, y_just_dict = create_just_data(x_test, y_test, x_just_name="x_test_just_", y_just_name="y_test_just_")

    name_of_x_test_just_sets = list(x_just_dict.keys())
    name_of_y_test_just_sets = list(y_just_dict.keys())
    name_of_models = list(model_dict.keys())
    name_of_criterions = list(criterion_dict.keys())

    for i in range(len(name_of_models)):
        model = model_dict[name_of_models[i]]
        criterion = criterion_dict[name_of_criterions[i]]

        accuracy_rec.loc[i, "model_name"] = name_of_models[i]

        for j in range(10):
            x_test_just = x_just_dict[name_of_x_test_just_sets[j]]
            y_test_just = y_just_dict[name_of_y_test_just_sets[j]]

            test_ds_just = TensorDataset(x_test_just, y_test_just)
            test_dl_just = DataLoader(test_ds_just, batch_size=batch_size * 2)

            test_loss, test_accuracy = validation(model, test_dl_just, criterion, device)

            accuracy_rec.loc[i, label_names[j]] = test_accuracy
    #             print( name_of_models[i], ">>" ,j, " tahmin etmesi: {:7.4f}".format(test_accuracy))
    #         print("******************")
    return accuracy_rec


def get_averaged_weights_faster(model_dict, device):
    name_of_models = list(model_dict.keys())
    parameters = list(model_dict[name_of_models[0]].named_parameters())
    ##named_parameters layer adını ve datayı tuple olarak dönderiyor
    ##parameters sadece datayı dönderiyor

    weight_dict = dict()
    for k in range(len(parameters)):
        name = parameters[k][0]
        w_shape = list(parameters[k][1].shape)
        w_shape.insert(0, len(model_dict))
        weight_info = torch.zeros(w_shape, device=device)
        weight_dict.update({name: weight_info})

    weight_names_list = list(weight_dict.keys())
    with torch.no_grad():
        for i in range(len(model_dict)):
            sample_param_data_list = list(model_dict[name_of_models[i]].parameters())
            for j in range(len(weight_names_list)):
                weight_dict[weight_names_list[j]][i,] = sample_param_data_list[j].data.clone()

        mean_weight_array = []
        for m in range(len(weight_names_list)):
            mean_weight_array.append(torch.mean(weight_dict[weight_names_list[m]], 0))

    return mean_weight_array


def set_averaged_weights_as_main_model_weights_and_update_main_model(main_model, model_dict, device):
    mean_weight_array = get_averaged_weights_faster(model_dict, device)
    main_model_param_data_list = list(main_model.parameters())
    with torch.no_grad():
        for j in range(len(main_model_param_data_list)):
            main_model_param_data_list[j].data = mean_weight_array[j]
    return main_model


def get_coordinate_wise_median_of_weights(model_dict, device):
    name_of_models = list(model_dict.keys())
    parameters = list(model_dict[name_of_models[0]].named_parameters())
    ##named_parameters layer adını ve datayı tuple olarak dönderiyor
    ##parameters sadece datayı dönderiyor

    weight_dict = dict()
    for k in range(len(parameters)):
        name = parameters[k][0]
        w_shape = list(parameters[k][1].shape)
        w_shape.insert(0, len(model_dict))
        weight_info = torch.zeros(w_shape, device=device)
        weight_dict.update({name: weight_info})

    weight_names_list = list(weight_dict.keys())
    with torch.no_grad():
        for i in range(len(model_dict)):
            sample_param_data_list = list(model_dict[name_of_models[i]].parameters())
            for j in range(len(weight_names_list)):
                weight_dict[weight_names_list[j]][i,] = sample_param_data_list[j].data.clone()

        median_weight_array = []
        for m in range(len(weight_names_list)):
            median_weight_array.append(torch.median(weight_dict[weight_names_list[m]], 0).values)

    return median_weight_array

def set_coordinatewise_med_weights_as_main_model_weights_and_update_main_model(main_model, model_dict, device):
    median_weight_array = get_coordinate_wise_median_of_weights(model_dict, device)
    main_model_param_data_list = list(main_model.parameters())
    with torch.no_grad():
        for j in range(len(main_model_param_data_list)):
            main_model_param_data_list[j].data = median_weight_array[j]
    return main_model








def get_averaged_weights_without_outliers_strict_condition(model_dict, iteration_distance, device):
    chosen_clients = iteration_distance[iteration_distance["include_calculation"] == True].index
    name_of_models = list(model_dict.keys())
    parameters = list(model_dict[name_of_models[0]].named_parameters())

    ### mesela conv 1 için zeros [chosen client kadar, 32, 1, 5, 5] atanıyor bunları doldurup mean alacağız
    weight_dict = dict()
    for k in range(len(parameters)):
        name = parameters[k][0]
        w_shape = list(parameters[k][1].shape)
        w_shape.insert(0, len(chosen_clients))
        weight_info = torch.zeros(w_shape, device=device)
        weight_dict.update({name: weight_info})

    weight_names_list = list(weight_dict.keys())
    with torch.no_grad():
        for i in range(len(chosen_clients)):
            sample_param_data_list = list(model_dict[name_of_models[chosen_clients[i]]].parameters())
            for j in range(len(weight_names_list)):
                weight_dict[weight_names_list[j]][i,] = sample_param_data_list[j].data.clone()

        mean_weight_array = []
        for m in range(len(weight_names_list)):
            mean_weight_array.append(torch.mean(weight_dict[weight_names_list[m]], 0))

    return mean_weight_array


def strict_condition_without_outliers_set_averaged_weights_as_main_model_weights_and_update_main_model(main_model,
                                                                                                       model_dict,
                                                                                                       iteration_distance,
                                                                                                       device):
    mean_weight_array = get_averaged_weights_without_outliers_strict_condition(model_dict, iteration_distance, device)
    main_model_param_data_list = list(main_model.parameters())
    with torch.no_grad():
        for j in range(len(main_model_param_data_list)):
            main_model_param_data_list[j].data = mean_weight_array[j]
    return main_model

## they do not perform any training and send same parameters that are received at the beginning at the fl round
def start_train_end_node_process_with_anticatalysts(number_of_samples, x_train_dict, y_train_dict, x_test_dict, y_test_dict,
                                                    batch_size, model_dict, criterion_dict, optimizer_dict,
                                                    numEpoch, byzantine_node_list, device):
    name_of_x_train_sets = get_x_train_sets_names(x_train_dict)
    name_of_y_train_sets = get_y_train_sets_names(y_train_dict)
    name_of_x_test_sets = get_x_test_sets_names(x_test_dict)
    name_of_y_test_sets = get_y_test_sets_names(y_test_dict)
    name_of_models = get_model_names(model_dict)
    name_of_criterions = get_criterion_names(criterion_dict)
    name_of_optimizers = get_optimizer_names(optimizer_dict)



    trusted_nodes=  np.array(list(set(np.arange(number_of_samples)) - set(byzantine_node_list)), dtype=int)

    ## STANDARD LOCAL MODEL TRAİNİNG PROCESS FOR TRUSTED NODES
    for i in trusted_nodes:

        train_ds = TensorDataset(x_train_dict[name_of_x_train_sets[i]], y_train_dict[name_of_y_train_sets[i]])
        train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

        test_ds = TensorDataset(x_test_dict[name_of_x_test_sets[i]], y_test_dict[name_of_y_test_sets[i]])
        test_dl = DataLoader(test_ds, batch_size=batch_size * 2)

        model = model_dict[name_of_models[i]]
        criterion = criterion_dict[name_of_criterions[i]]
        optimizer = optimizer_dict[name_of_optimizers[i]]

        for epoch in range(numEpoch):
            train_loss, train_accuracy = train(model, train_dl, criterion, optimizer, device)
            test_loss, test_accuracy = validation(model, test_dl, criterion, device)


######################################
#### a little is enough functions
def get_zscore_for_a_little_is_enough (number_of_samples,hostile_node_percentage):
#     from statistics import NormalDist is defined at the top
    malicious = int(number_of_samples * hostile_node_percentage)
    supporter = np.floor((number_of_samples / 2) + 1) - malicious
    area = (number_of_samples - malicious - supporter) / (number_of_samples - malicious)
    zscore = NormalDist().inv_cdf(area)
    return zscore
def get_byzantine_node_stats_for_a_little(model_dict,byzantine_node_list, device):
    name_of_models = []
    for node in byzantine_node_list:
        name_of_models.append("model"+str(node))
    parameters = list(model_dict[name_of_models[0]].named_parameters())
    ##named_parameters layer adını ve datayı tuple olarak dönderiyor
    ##parameters sadece datayı dönderiyor
    weight_dict = dict()
    for k in range(len(parameters)):
        name = parameters[k][0]
        w_shape = list(parameters[k][1].shape)
        w_shape.insert(0, len(byzantine_node_list))
        weight_info = torch.zeros(w_shape, device=device)
        weight_dict.update({name: weight_info})
    weight_names_list = list(weight_dict.keys())
    with torch.no_grad():
        for i in range(len(byzantine_node_list)):
            sample_param_data_list = list(model_dict[name_of_models[i]].parameters())
            for j in range(len(weight_names_list)):
                weight_dict[weight_names_list[j]][i,] = sample_param_data_list[j].data.clone()
        mean_weight_array = []
        std_weight_array = []
        for m in range(len(weight_names_list)):
            mean_weight_array.append(torch.mean(weight_dict[weight_names_list[m]], 0))
            std_weight_array.append(torch.std(weight_dict[weight_names_list[m]], 0))
    return mean_weight_array,std_weight_array
def change_parameters_of_hostile_nodes(model_dict, byzantine_node_list,zscore, device):
    name_of_models = list(model_dict.keys())
    with torch.no_grad():
        mean_weight_array,std_weight_array = get_byzantine_node_stats_for_a_little(model_dict,byzantine_node_list, device=device)
        for j in byzantine_node_list:
            hostile_node_param_data_list = list(model_dict[name_of_models[j]].parameters())
            for k in range(len(hostile_node_param_data_list)):
                hostile_node_param_data_list[k].data= mean_weight_array[k]-std_weight_array[k]*zscore
            model_dict[name_of_models[j]].float()
    return model_dict
#####################################
def get_trimmed_mean(model_dict, hostile_node_percentage, device):
    name_of_models = list(model_dict.keys())
    parameters = list(model_dict[name_of_models[0]].named_parameters())
    weight_dict = dict()
    for k in range(len(parameters)):
        name = parameters[k][0]
        w_shape = list(parameters[k][1].shape)
        w_shape.insert(0, len(model_dict))
        weight_info = torch.zeros(w_shape, device=device)
        weight_dict.update({name: weight_info})
    weight_names_list = list(weight_dict.keys())
    with torch.no_grad():
        for i in range(len(model_dict)):
            sample_param_data_list = list(model_dict[name_of_models[i]].parameters())
            for j in range(len(weight_names_list)):
                weight_dict[weight_names_list[j]][i,] = sample_param_data_list[j].data.clone()
    mean_weight_array = []
    for m in range(len(weight_names_list)):
        layers_from_nodes = weight_dict[weight_names_list[m]]
        trim_layer_info=stats.trim_mean(layers_from_nodes.clone().cpu(), hostile_node_percentage, axis=0)
        mean_weight_array.append(trim_layer_info)
    return mean_weight_array
def set_trimmed_mean_weights_as_main_model_weights_and_update_main_model(main_model, model_dict, hostile_node_percentage, device):
    mean_weight_array = get_trimmed_mean(model_dict, hostile_node_percentage, device)
    main_model_param_data_list = list(main_model.parameters())
    with torch.no_grad():
        for j in range(len(main_model_param_data_list)):
                        main_model_param_data_list[j].data = torch.tensor(mean_weight_array[j], dtype=torch.float32, device=device)
    return main_model
######################################
# fang partial knowledge attack adaptation
def partial_knowledge_fang_ind(main_model, model_dict,byzantine_node_list, iteration_byzantine_seed, device):
    name_of_models = list(model_dict.keys())
    with torch.no_grad():
        mean_weight_array, std_weight_array = get_byzantine_node_stats_for_a_little(model_dict, byzantine_node_list,
                                                                                       device=device)
        main_model_param_data_list = list(main_model.parameters())
        for j in byzantine_node_list:
            hostile_node_param_data_list = list(model_dict[name_of_models[j]].parameters())
            for k in range(len(hostile_node_param_data_list)):
                original_shape = list(hostile_node_param_data_list[k].data.shape)
                data = np.zeros(original_shape)
                hostile_data = hostile_node_param_data_list[k].data.clone().data.cpu()
                main_model_data = main_model_param_data_list[k].data.clone().data.cpu()
                mean = mean_weight_array[k].clone().data.cpu()
                std = std_weight_array[k].clone().data.cpu()
                sign_matrix = (hostile_data > main_model_data)
                np.random.seed(iteration_byzantine_seed) ## nodelar kendi mean stdlerine göre alıyor her experimentin x. roundı aynı gibi
                data[sign_matrix == True] = np.random.uniform(
                    low=mean[sign_matrix == True] - 4 * std[sign_matrix == True],
                    high=mean[sign_matrix == True] - 3 * std[sign_matrix == True])
                np.random.seed(iteration_byzantine_seed)
                data[sign_matrix == False] = np.random.uniform(
                    low=mean[sign_matrix == False] + 3 * std[sign_matrix == False],
                    high=mean[sign_matrix == False] + 4 * std[sign_matrix == False])
                hostile_node_param_data_list[k].data = torch.tensor(data,dtype=torch.float32, device=device)
            model_dict[name_of_models[j]].float()
    return model_dict
def partial_knowledge_fang_org(main_model, model_dict, byzantine_node_list, iteration_byzantine_seed, device):
    name_of_models = list(model_dict.keys())
    with torch.no_grad():
        mean_weight_array, std_weight_array = get_byzantine_node_stats_for_a_little(model_dict, byzantine_node_list,
                                                                                       device=device)
        main_model_param_data_list = list(main_model.parameters())
        organized = []
        for k in range(len(main_model_param_data_list)):
            original_shape = list(main_model_param_data_list[k].data.shape)
            data = np.zeros(original_shape)
            main_model_data = main_model_param_data_list[k].clone().data.cpu()
            mean = mean_weight_array[k].clone().data.cpu()
            std = std_weight_array[k].clone().data.cpu()
            sign_matrix = (mean > main_model_data)
            np.random.seed(iteration_byzantine_seed)
            data[sign_matrix == True] = np.random.uniform(
                low=mean[sign_matrix == True] - 4 * std[sign_matrix == True],
                high=mean[sign_matrix == True] - 3 * std[sign_matrix == True])
            np.random.seed(iteration_byzantine_seed)
            data[sign_matrix == False] = np.random.uniform(
                low=mean[sign_matrix == False] + 3 * std[sign_matrix == False],
                high=mean[sign_matrix == False] + 4 * std[sign_matrix == False])
            organized.append(data)
    for b in byzantine_node_list:
        hostile_node_param_data_list = list(model_dict[name_of_models[b]].parameters())
        for m in range(len(hostile_node_param_data_list)):
            hostile_node_param_data_list[m].data = torch.tensor(organized[m], dtype=torch.float32, device=device)
        model_dict[name_of_models[b]].float()
    return model_dict