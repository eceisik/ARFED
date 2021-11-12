import torch
from torch import nn
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import numpy as np
from fl_utils import distribute_data as dd
from fl_utils import train_nodes as tn
from fl_utils import construct_models as cm
from torch.optim import lr_scheduler

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("device: ", device)

number_of_samples = 100 #number of participants

is_noniid = True
if is_noniid:
    n = 5
    min_n_each_node = 5
else:
    n = 10
    min_n_each_node = 10

is_organized = True
hostile_node_percentage = 0.20 #malicious participant ratio
byzantine_mean = 0
byzantine_std = 1

iteration_num = 500 #number of communication rounds
learning_rate = 0.0015
min_lr = 0.000010
lr_scheduler_factor = 0.2
best_threshold = 0.0001
clipping = True
clipping_threshold = 10

weight_decay = 0.0001
numEpoch = 10
batch_size = 100
momentum = 0.9

seed = 7
use_seed = 17
hostility_seed = 33
converters_seed = 221
byzantine_seed = 96
factor = 1.5

train_amount = 5000
test_amount = 1000


x_train, y_train, x_test, y_test = dd.load_cifar_data()

##train
label_dict_train = dd.split_and_shuffle_labels(y_data=y_train, seed=seed, amount=train_amount)
node_label_info_train, total_label_occurences_train, amount_info_table_train = dd.get_info_for_distribute_non_iid_with_different_n_and_amount(
    number_of_samples=number_of_samples, n=n, amount=train_amount, seed=use_seed, min_n_each_node=min_n_each_node)

x_train_dict, y_train_dict = dd.distribute_cifar_data_to_participants(label_dict=label_dict_train,
                                                                      amount=train_amount,
                                                                      number_of_samples=number_of_samples,
                                                                      n=n, x_data=x_train,
                                                                      y_data=y_train,
                                                                      node_label_info=node_label_info_train,
                                                                      amount_info_table=amount_info_table_train,
                                                                      x_name="x_train",
                                                                      y_name="y_train")

## test
label_dict_test = dd.split_and_shuffle_labels(y_data=y_test, seed=seed, amount=test_amount)
node_label_info_test, total_label_occurences_test, amount_info_table_test = dd.get_info_for_distribute_non_iid_with_different_n_and_amount(
    number_of_samples=number_of_samples,
    n=n, amount=test_amount, seed=use_seed, min_n_each_node=min_n_each_node)
x_test_dict, y_test_dict = dd.distribute_cifar_data_to_participants(label_dict=label_dict_test,
                                                                    amount=test_amount,
                                                                    number_of_samples=number_of_samples,
                                                                    n=n, x_data=x_test,
                                                                    y_data=y_test,
                                                                    node_label_info=node_label_info_test,
                                                                    amount_info_table=amount_info_table_test,
                                                                    x_name="x_test",
                                                                    y_name="y_test")

train_ds = TensorDataset(x_train, y_train)
train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

test_ds = TensorDataset(x_test, y_test)
test_dl = DataLoader(test_ds, batch_size=batch_size * 2)

main_model = cm.Cifar10CNN()
cm.weights_init(main_model)
main_model = main_model.to(device)

main_optimizer = torch.optim.SGD(main_model.parameters(), lr=learning_rate, momentum=momentum,
                                 weight_decay=weight_decay)
main_criterion = nn.CrossEntropyLoss()
scheduler = lr_scheduler.ReduceLROnPlateau(main_optimizer, mode="max", factor=lr_scheduler_factor,
                                           patience=10, threshold=best_threshold, verbose=True, min_lr=min_lr)

model_dict, optimizer_dict, criterion_dict = tn.create_model_optimizer_criterion_dict_for_cifar_cnn(number_of_samples,
                                                                                                    learning_rate,
                                                                                                    momentum, device,
                                                                                                    weight_decay)

test_accuracies_of_each_iteration = np.array([], dtype=float)

byzantine_node_list = dd.get_byzantine_node_list(hostile_node_percentage, number_of_samples, hostility_seed)
np.random.seed(byzantine_seed)
byzantine_seeds_array = np.random.choice(5000, size=iteration_num, replace=False)

for iteration in range(iteration_num):

    model_dict = tn.send_main_model_to_nodes_and_update_model_dict(main_model, model_dict, number_of_samples)

    if is_organized:
        iteration_byzantine_seed = byzantine_seeds_array[iteration]
    else:
        iteration_byzantine_seed = None

    tn.start_train_end_node_process_byzantine_for_cifar_with_augmentation(number_of_samples, x_train_dict, y_train_dict,
                                                                          x_test_dict, y_test_dict,
                                                                          batch_size, model_dict, criterion_dict,
                                                                          optimizer_dict,
                                                                          numEpoch, byzantine_node_list, byzantine_mean,
                                                                          byzantine_std,
                                                                          device, clipping, clipping_threshold,
                                                                          iteration_byzantine_seed)


    main_model = tn.set_averaged_weights_as_main_model_weights_and_update_main_model(main_model, model_dict, device)
    test_loss, test_accuracy = tn.validation(main_model, test_dl, main_criterion, device)
    scheduler.step(test_accuracy)
    new_lr = main_optimizer.param_groups[0]["lr"]
    optimizer_dict = cm.update_learning_rate_decay(optimizer_dict, new_lr)

    test_accuracies_of_each_iteration = np.append(test_accuracies_of_each_iteration, test_accuracy)
    print("Iteration", str(iteration + 1), ": main_model accuracy on all test data: {:7.4f}".format(test_accuracy))




