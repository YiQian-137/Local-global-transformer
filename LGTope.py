import time
import os
import torch
import torch.nn as nn
from torch import optim
from torch.optim import lr_scheduler
from model import TransformerPE
from utils import get_train_data, get_test_data
import scipy.io as sio
import matplotlib.pyplot as plt
import numpy as np
import random
import torch.nn.functional as F

def train():
    model = TransformerPE()
    model.to(device)
    Epoch = 1800
    optimizer = optim.Adam(model.parameters(), lr=0.002, betas=(0.9, 0.99), eps=1e-08, weight_decay=0.0001)  # alt+enter
    scheduler = lr_scheduler.MultiStepLR(optimizer, [500], 0.5)
    criterion = nn.MSELoss()
    model.train()
    epoches = []
    train_loss = []
    test_loss = []

    for epoch in range(Epoch):
        running_loss = 0.0
        optimizer.zero_grad()
        decoder_out, out, _ = model(train_data_unit1, train_data_unit2, train_data_unit3, train_data_unit4,train_data_unit5)
        loss = criterion(decoder_out, train_data)
        loss.backward()
        optimizer.step()
        scheduler.step()
        running_loss += loss.item()

        if (epoch + 1) % 50 == 0:
            print('epoch_{}'.format(epoch+1))
            print('training_loss:{}'.format(running_loss))
            epoches.append(epoch)
            train_loss.append(running_loss)
            #保存模型参数
            base_path = "./TE/model/"
            if not os.path.exists(base_path):
                os.makedirs(base_path)
            torch.save(model.state_dict(),os.path.join(base_path, "transformer_para_{}".format(epoch + 1)))
            # model.load_state_dict(
            #     torch.load(os.path.join(base_path, "transformer_para_{}".format(epoch + 1))))
            
            model.eval()
            running_loss_0 = 0.0
            with torch.no_grad():
                decoder_out, out, hidden_state = model(test_data_unit1, test_data_unit2, test_data_unit3, test_data_unit4, test_data_unit5)
                loss = criterion(decoder_out, test_data)
                running_loss_0 += loss.item()
                test_loss.append(running_loss_0)
                print('testing_loss:{}'.format(loss))
                
            global_fea = out.detach().cpu().numpy()
            local_fea = hidden_state.detach().cpu().numpy()
            #print(local_fea.shape),(21120,128)
            mat1 = {'fea_test': global_fea.T}
            mat2 = {'fea_test': local_fea.T}
            save_path = "./TE/feature/"
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            sio.savemat(os.path.join(save_path,"fea_glo_{}.mat".format(epoch + 1)), mat1)
            sio.savemat(os.path.join(save_path,"fea_loc_{}.mat".format(epoch + 1)), mat2)
            # print("testing is finished!")

    print("time span: %.4f s" % (time.time() - start))
    # plt.plot(epoches, train_loss, color='r', label='train loss')
    plt.figure()
    plt.plot(epoches, test_loss, color='b', label='test loss')
    plt.figure()
    plt.plot(epoches, train_loss, color='r', label='train_loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(loc="best")
    plt.show()


def test():
    model = TransformerPE()
    model.to(device)

    base_path = "./TE/model/"
    model.load_state_dict(
        torch.load(os.path.join(base_path, "transformer_para_{}".format(1800))))
    model.eval()

    with torch.no_grad():
        decoder_out, out, hidden_state = model(test_data_unit1, test_data_unit2, test_data_unit3, test_data_unit4, test_data_unit5)
        n = decoder_out - test_data
 
        
    global_fea = out.detach().cpu().numpy()
    local_fea = hidden_state.detach().cpu().numpy()
    res = n.detach().cpu().numpy()

    mat1 = {'fea_test': global_fea.T}
    mat2 = {'fea_test': local_fea.T}
    mat3 = {'res_test': res.T}


    save_path = "./TE/feature/222/"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    sio.savemat(os.path.join(save_path,"fea_glo.mat"), mat1)
    sio.savemat(os.path.join(save_path,"fea_loc.mat"), mat2)
    sio.savemat(os.path.join(save_path,"res.mat"), mat3)
  
    print('finishing!')

if __name__ == '__main__':
    start = time.time()

    unit1 = [0, 1, 2, 3, 41, 42, 43, 44]
    unit2 = [5, 6, 7, 8, 20, 50, 51]
    unit3 = [10, 11, 12, 13, 21, 47]
    unit4 = [14, 15, 16, 17, 18, 48, 49]
    unit5 = [4, 9, 19, 45, 46]
    unit_list = [unit1, unit2, unit3, unit4, unit5]
    unit_variables = unit1+unit2+unit3+unit4+unit5
    unit_variables = [i+1 for i in unit_variables]
    unit_len = [8, 7, 6, 7, 5]

    train_data_unit1 = get_train_data(unit1)
    train_data_unit1 = torch.from_numpy(train_data_unit1)
    train_data_unit2 = get_train_data(unit2)
    train_data_unit2 = torch.from_numpy(train_data_unit2)
    train_data_unit3 = get_train_data(unit3)
    train_data_unit3 = torch.from_numpy(train_data_unit3)
    train_data_unit4 = get_train_data(unit4)
    train_data_unit4 = torch.from_numpy(train_data_unit4)
    train_data_unit5 = get_train_data(unit5)
    train_data_unit5 = torch.from_numpy(train_data_unit5)
    train_data = torch.cat((train_data_unit1, train_data_unit2, train_data_unit3, train_data_unit4, train_data_unit5),dim=1)

    test_data_unit1 = get_test_data(unit1)
    test_data_unit1 = torch.from_numpy(test_data_unit1)
    test_data_unit2 = get_test_data(unit2)
    test_data_unit2 = torch.from_numpy(test_data_unit2)
    test_data_unit3 = get_test_data(unit3)
    test_data_unit3 = torch.from_numpy(test_data_unit3)
    test_data_unit4 = get_test_data(unit4)
    test_data_unit4 = torch.from_numpy(test_data_unit4)
    test_data_unit5 = get_test_data(unit5)
    test_data_unit5 = torch.from_numpy(test_data_unit5)
    test_data = torch.cat((test_data_unit1, test_data_unit2, test_data_unit3, test_data_unit4, test_data_unit5),dim=1)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    train_data = train_data.to(device)
    train_data_unit1 = train_data_unit1.to(device)
    train_data_unit2 = train_data_unit2.to(device)
    train_data_unit3 = train_data_unit3.to(device)
    train_data_unit4 = train_data_unit4.to(device)
    train_data_unit5 = train_data_unit5.to(device)

    test_data = test_data.to(device)
    test_data_unit1 = test_data_unit1.to(device)
    test_data_unit2 = test_data_unit2.to(device)
    test_data_unit3 = test_data_unit3.to(device)
    test_data_unit4 = test_data_unit4.to(device)
    test_data_unit5 = test_data_unit5.to(device)

    train()
    test()
