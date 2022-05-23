import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import copy
import numpy as np
from torchvision import datasets, transforms
import torch
#from torch.utils.tensorboard import SummaryWriter
from utils.sampling import mnist_iid, mnist_noniid, cifar_iid
from utils.synthetic import *
from utils.options import args_parser
from models.Update import *
from models.Nets import MLP, CNNMnist, CNNCifar, logistic_regression, RNNModel
from models.Aggregation import FedAvg, FedAvg_sample, FedAvgPers
from models.test import test_img
from utils.partition_data import generate_power_dataset
from utils.language_utils import *
import random
import datetime
from datetime import date
import pickle
from torch.optim.lr_scheduler import MultiStepLR

def init_net(args):
    if args.model == "cifarCNN" and args.dataset == "cifar10":
        net = CNNCifar(args).to(args.device)
    elif args.model == 'logistic' and args.dataset == 'synthetic':
        net = logistic_regression(input_dim = 60, output_dim =10).to(args.device)
    elif args.model == 'mlp' and args.dataset == 'mnist':
        net = MLP(dim_hidden = 128, dim_in = 784, dim_out = 10).to(args.device)
    elif args.model == "rnn" and args.dataset == "shakespeare":
        net = RNNModel('LSTM', 80, 8, 256, 2, 0.0, tie_weights=False).to(args.device)
    else:
        exit('Error: unrecognized model')
    return net

if __name__ == '__main__':
    
    
    #log_dir = "runs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    #log_dir = "runs/" + date.today().strftime("%Y%m%d")
    #writer = SummaryWriter(log_dir)
    args = args_parser()

    # set seed
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')

    # load dataset and split users
    if args.dataset == 'mnist' or args.dataset == "cifar10":
        if args.read_partition == True:
            p_file = 'p_{}_TN{}_sk{}_C{}_s{}'.format(args.dataset, args.num_users, args.skewness_factor, args.classes_per_partition, 1)
            p_file = "Partitions/" + p_file + ".p"
            dataset_train, dataset_test, trainDataset_local , testDataset_local, partition_stats = pickle.load(open(p_file,"rb"))
        else:
            dataset_train, dataset_test, trainDataset_local , testDataset_local, partition_stats = generate_power_dataset(args)
    elif args.dataset == "shakespeare":
        dataset_train, dataset_test, trainDataset_local, testDataset_local, partition_stats = generate_shake()
    elif args.dataset == "synthetic":
        if args.read_partition == True:
            p_file = 'p_{}_TN{}_iid_{}_s{}'.format(args.dataset, args.num_users, args.iid, 1)
            p_file = "Partitions/" + p_file + ".p"
            dataset_train, dataset_test, trainDataset_local , testDataset_local, partition_stats = pickle.load(open(p_file,"rb"))
        else:
            dataset_train, dataset_test, trainDataset_local , testDataset_local, partition_stats = generate_synthetic_datasets(args)
    else:
        exit('Error: unrecognized dataset')
    img_size = dataset_train[0][0].shape

    if args.model == "cifarCNN" and args.dataset == "cifar10":
        net_glob = CNNCifar(args).to(args.device)
    elif args.model == 'logistic' and args.dataset == 'synthetic':
        net_glob = logistic_regression(input_dim = 60, output_dim =10).to(args.device)
    elif args.model == 'mlp' and args.dataset == 'mnist':
        net_glob = MLP(dim_hidden = 128, dim_in = 784, dim_out = 10).to(args.device)
    elif args.model == "rnn" and args.dataset == "shakespeare":
        # no dropout 
        net_glob = RNNModel('LSTM', 80, 8, 256, 2, 0.0, tie_weights=False).to(args.device)
    else:
        exit('Error: unrecognized model')
    print(net_glob)
    net_glob.train()

    num_users = len(trainDataset_local)
    w_glob = net_glob.state_dict()
    
    # Initializations for scaffold
    c_global = init_net(args)
    c_global = list(c_global.parameters()) 
    c_all = [copy.deepcopy(c_global) for i in range(num_users)]
    
    lamb_track= []
    gamma_track = []
    sample_weight_track = []
    idxs_track = []

    lamb = torch.zeros(num_users, requires_grad=True,device="cuda")
    if args.fix_gamma:
        opt_lamb = torch.optim.SGD([lamb], lr=args.lamb_lr, momentum=args.lamb_momentum)  
    else:
        gamma = torch.zeros(num_users,  requires_grad=True,device="cuda")
        opt_lamb = torch.optim.SGD([lamb,gamma], lr=args.lamb_lr, momentum=args.lamb_momentum)  
    
    print(args.dataset)
    print(args.device)
    
    # global model evaluated on the entire dataset
    global_acc_track = []
    global_loss_track = []
    # global model evaluated on the local dataset
    local_acc_track = []
    local_loss_track = []
    local_acc_per_track = []

    w_locals_all = [w_glob for i in range(args.num_users)]

    for iter in range(1, args.epochs+1):
        w_locals = []
        
        # User sampling
        idxs_users = np.random.choice(range(num_users), args.num_local_users, replace=False)
        lenData_local = []
        hs = []
        deltas_all = []
        delta_c_all = []
        total_data = 0.

        # Local training
        for idx in idxs_users:
            local = LocalUpdate(args=args, train_dataset=trainDataset_local[idx], test_dataset= testDataset_local[idx])

            lamb_temp = copy.deepcopy(lamb[idx].item())
            if args.fix_gamma:
                gamma_temp = args.gamma
            else:
                gamma_temp = copy.deepcopy(gamma[idx].item())
            
            if args.alg == "fedavg" or args.alg == "fedprox" or args.alg  == "fedbc":
                if args.train_MAML:
                    w = local.train_MAML(copy.deepcopy(net_glob).to(args.device), net_glob, lamb_temp, gamma_temp) 
                else:
                    w = local.train(copy.deepcopy(net_glob).to(args.device), net_glob, lamb_temp, gamma_temp)
            elif args.alg == "pfedme":
                    w = local.train_pfedme(copy.deepcopy(net_glob).to(args.device), copy.deepcopy(net_glob).to(args.device), lamb_temp, gamma_temp) 
            elif args.alg == "qfedavg":
                loss_local = local.evaluate_loss(net_glob, trainDataset_local[idx])
                w, delta_w = local.train_qfedavg(copy.deepcopy(net_glob).to(args.device), net_glob)
                norm_sum = 0
                deltas = []  
                for i, dw in enumerate(delta_w):
                    delta_temp = ((loss_local + 1e-10) ** args.q) * (1. / args.lr) * dw 
                    deltas.append(delta_temp)     
                    norm_sum = norm_sum + (dw.square().sum()) 
                h_temp = args.q * ((loss_local + 1e-10) ** (args.q - 1)) * norm_sum + (1. / args.lr) * (loss_local ** args.q)
                hs.append(h_temp)
                deltas_all.append(deltas)
            elif args.alg == "scaffold":
                w, c, delta_c = local.train_scaffold(copy.deepcopy(net_glob).to(args.device), net_glob, c_all[idx], c_global)
                c_all[idx] = c
                delta_c_all.append(delta_c)
                
            lenData_local.append(len(trainDataset_local[idx]))
            total_data = total_data + len(trainDataset_local[idx]) 
            w_locals_all[idx] = copy.deepcopy(w)
            w_locals.append(copy.deepcopy(w))
        
            # Update lamb (dual ascent)
            if args.alg == "fedbc":
                opt_lamb.zero_grad()
                param_diff = 0.
                for w_g, wk in zip(net_glob.parameters(), w.keys()):
                    param_diff = param_diff + ((w_g.data - w[wk]).square().sum()) 
                if args.fix_gamma:
                    lamb_loss = (param_diff - args.gamma) * lamb[idx]
                else:
                    lamb_loss = (param_diff - gamma[idx]) * lamb[idx]
                lamb_loss.backward()
                lamb.grad.data = -lamb.grad.data
                opt_lamb.step()
                # ensure lamb greater than zero
                with torch.no_grad():
                    lamb.clamp_(min=0)
        
        lamb_track.append(lamb.detach().cpu().numpy())
        if args.fix_gamma:
            gamma_track.append(args.gamma)
        else:
            gamma_track.append(gamma.detach().cpu().numpy())
        sample_weight_track.append(lenData_local)
        idxs_track.append(idxs_users)

        # Aggregation
        if args.alg == "fedavg":   
            w_glob = FedAvg_sample(w_locals,np.array(lenData_local))
            net_glob.load_state_dict(w_glob)
        elif args.alg == "fedprox":
            w_glob = FedAvg_sample(w_locals,np.array(lenData_local))
            net_glob.load_state_dict(w_glob)
        elif args.alg ==  "fedbc": 
            lamb_temp = lamb.detach().cpu().numpy()
            lamb_round = lamb_temp[idxs_users]
            w_glob = FedAvgPers(w_locals, lamb_round)   
            net_glob.load_state_dict(w_glob) 
        elif args.alg == "pfedme":
            w_glob = FedAvg_sample(w_locals, lenData_local)
            net_glob.load_state_dict(w_glob)
        elif args.alg ==  "qfedavg": 
            h_final = 0.
            for i in range(len(idxs_users)): 
                if i == 0:
                    wdiff_finals = []
                    for j in range(len(deltas_all[i])):
                        wdiff_finals.append(lenData_local[i] * deltas_all[i][j])
                else:
                    for j in range(len(deltas_all[i])):
                        wdiff_finals[j] = wdiff_finals[j] + lenData_local[i] * deltas_all[i][j] 
                h_final = h_final + lenData_local[i] * hs[i]                 

            for i, (w, d) in enumerate(zip(net_glob.parameters(), wdiff_finals)):
                w.data = w.data - d / h_final

        elif args.alg == "scaffold":
            for i in range(len(idxs_users)):
                if i == 0:
                    c_delta_final = []
                    for j in range(len(delta_c_all[i])):
                        c_delta_final.append(delta_c_all[i][j])
                else:
                    for j in range(len(delta_c_all[i ])):
                        c_delta_final[j] = c_delta_final[j] + delta_c_all[i][j] 
            for j in range(len(c_delta_final)):
                c_delta_final[j] = c_delta_final[j] / num_users
                c_global[j] = c_global[j] + c_delta_final[j] 
            
            w_glob = FedAvg_sample(w_locals,np.array(lenData_local))
            net_glob.load_state_dict(w_glob)

        # Evaluate local model performance
        loss_train_pers = 0.
        correct_pers = 0.
        acc_test_pers = 0.
        total_num_test_data = 0.
        total_num_train_data = 0.
        num_train_data_temp = []
        num_test_data_temp = []
        if args.eval_local == True:
            
            acc_test_local = []
            for i in range(num_users):
                if args.test_MAML:
                    net_temp_local = copy.deepcopy(net_glob)
                else:             
                    net_temp_local = copy.deepcopy(net_glob)
                    net_temp_local.load_state_dict(w_locals_all[i])

                train_dataset_temp = trainDataset_local[i]
                test_dataset_temp = testDataset_local[i]
                
                local_trainloader = DataLoader(train_dataset_temp, args.bs)
                local_testloader = DataLoader(test_dataset_temp, args.bs)

                evaluate_driver_local = evaluate_local(args, net_temp_local, test_dataset_temp)

                if args.eval_one_step:
                    evaluate_driver_local.train_one_step()
                
                if args.dataset == "shakespeare":
                    _, local_correct_test, num_test_data = evaluate_driver_local.eval_nlp(local_testloader)
                    local_loss_train, _, num_train_data = evaluate_driver_local.eval_nlp(local_trainloader)
                else:
                    _, local_correct_test, num_test_data = evaluate_driver_local.eval(local_testloader)
                    local_loss_train, _, num_train_data = evaluate_driver_local.eval(local_trainloader)
                
                    num_train_data_temp.append(num_train_data)
                    num_test_data_temp.append(num_test_data)
                    
                acc_test_local.append(local_correct_test / num_test_data)
                loss_train_pers += local_loss_train
                correct_pers += local_correct_test
                total_num_test_data += num_test_data
                total_num_train_data += num_train_data

            loss_train_pers = loss_train_pers / total_num_train_data
            acc_test_pers = correct_pers / total_num_test_data * 100
            local_acc_per_track.append(acc_test_local)
            local_acc_track.append(acc_test_pers)
            local_loss_track.append(loss_train_pers)

        # Evaluate global model performance
        if args.dataset == "shakespeare":
            _, loss_train_global = test_nlp(net_glob, dataset_train, args)
            acc_test_global, _ = test_nlp(net_glob, dataset_test, args)
            if args.eval_local == False:
                acc_test_pers = acc_test_global
                loss_train_pers = loss_train_global
        else:
            _, loss_train_global = test_img(net_glob, dataset_train, args)
            acc_test_global, _ = test_img(net_glob, dataset_test, args)
            if args.eval_local == False:
                acc_test_pers = acc_test_global
                loss_train_pers = loss_train_global

        global_acc_track.append(acc_test_global)
        global_loss_track.append(loss_train_global)
        print('Global Round {:3d}, Train loss {:.3f}, Test acc {:.3f}'.format(iter, loss_train_global, acc_test_global))
        print('Local Round {:3d}, Train loss {:.3f}, Test acc {:.3f}'.format(iter, loss_train_pers, acc_test_pers))


    lamb_track = np.array(lamb_track)
    gamma_track = np.array(gamma_track)
    local_acc_per_track = np.array(local_acc_per_track)
    local_acc_track = np.array(local_acc_track)
    local_loss_track = np.array(local_loss_track)
    global_acc_track = np.array(global_acc_track)
    global_loss_track = np.array(global_loss_track)
    sample_weight_track = np.array(sample_weight_track)
    idxs_track = np.array(idxs_track)
    
    train_data_len = []
    test_data_len = []
    for i in range(num_users):
        train_data_len.append(len(trainDataset_local[i]))
        test_data_len.append(len(testDataset_local[i]))

    
    # Save results
    output_file = 'results_{}_{}_{}_{}_R{}_E{}_B{}_TB{}_TN{}_N{}_mu{}_ga{}_lr{}_lrL{}_innerlr{}_outerlr{}_MAML{}_iid{}_C{}_sk{}_q{}_K{}_adapt{}_testMAML{}_s{}_v2'.format(args.dataset, args.model, args.alg, \
                        args.alpha, args.epochs, args.local_ep, args.local_bs, args.bs, \
                        num_users, args.num_local_users, args.mu, \
                        args.gamma, args.lr, args.lamb_lr, \
                        args.inner_lr, args.outer_lr, args.train_MAML, args.iid, \
                        args.classes_per_partition, args.skewness_factor, args.q, args.K, args.eval_one_step, args.test_MAML, args.seed)
    print(output_file) 
    results_d = args.results_dir 
    if not os.path.isdir(results_d):
        os.mkdir(results_d)
    results_d2 = args.results_dir + "/" + args.alg
    if not os.path.isdir(results_d2):
        os.mkdir(results_d2)
    output_file =  results_d2 + "/" + output_file + ".p"
    results_final = {}
    results_final["lamb_track"] = lamb_track
    results_final["gamma_track"] = gamma_track
    results_final["local_acc_per_track"] = local_acc_per_track
    results_final["local_acc_track"] = local_acc_track
    results_final["local_loss_track"] = local_loss_track 
    results_final["global_acc_track"] = global_acc_track  
    results_final["global_loss_track"] = global_loss_track
    results_final["sample_weight_track"] = sample_weight_track
    results_final["idxs_track"] = idxs_track
    results_final["partition_stats"] = (train_data_len, test_data_len)
    pickle.dump(results_final, open(output_file, "wb" ))


