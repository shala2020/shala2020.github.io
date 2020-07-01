from GRUD_RNN import * 
from pandas import read_csv
import matplotlib.pyplot as plt

def PrepareDataset(lightCurve_matrix, \
                   BATCH_SIZE = 40, \
                   seq_len = 10, \
                   pred_len = 10, \
                   train_propotion = 0.5, \
                   valid_propotion = 0.1, \
                   mask_ones_proportion = 1.0):
    print('Split lightCurve finished.')
    totalDays = lightCurve_matrix.shape[0]

    max_lightCurve = lightCurve_matrix.max().max()
    lightCurve_matrix =  lightCurve_matrix / max_lightCurve
    lightCurve_sequences, lightCurve_labels = [], []
    for i in range(totalDays - seq_len - pred_len):
        lightCurve_sequences.append(lightCurve_matrix.iloc[i:i+seq_len].values)
        lightCurve_labels.append(lightCurve_matrix.iloc[i+seq_len:i+seq_len+pred_len].values)
    lightCurve_sequences, lightCurve_labels = np.asarray(lightCurve_sequences), np.asarray(lightCurve_labels)
    
    # using zero-one mask to randomly set elements to zeros
    
    print(' Start to generate Mask, Delta, Last_observed_X ...')
    
    # Mask M = 1 if X(t) is observed
    # Mask M = 0 if X(t) is not observed
    M = np.copy(lightCurve_sequences)       
    M[M != 0] = 1
     
    # temporal information
    interval = 1 # 1 day
    S = np.zeros_like(lightCurve_sequences) # time stamps
    for i in range(S.shape[1]):
        S[:,i,:] = interval * i
    
    #Delta = S(t) − S(t − 1) + δ(t−1) if t > 1, M(t−1) = 0 
    #Delta = S(t) − S(t − 1)          if t > 1, M(t−1) = 1 
    #Delta = 0                        if t =1 
    δ = np.zeros_like(lightCurve_sequences) # time intervals
    for i in range(1, S.shape[1]):
        δ[:,i,:] = S[:,i,:] - S[:,i-1,:]
        
    missing_index = np.where(M == 0)
        

    X_last_obsv = np.copy(lightCurve_sequences)
    for idx in range(missing_index[0].shape[0]):
        i = missing_index[0][idx] 
        j = missing_index[1][idx]
        k = missing_index[2][idx]
        if j != 0 and j != 9:
            δ[i,j+1,k] = δ[i,j+1,k] + δ[i,j,k]
        if j != 0:
            X_last_obsv[i,j,k] = X_last_obsv[i,j-1,k] # last observation
    δ = δ / δ.max() # normalize
       
    # Split the dataset to training and testing datasets
    print('Generate Mask, Delta, Last_observed_X finished. Start to split dataset ...')
    sample_size = lightCurve_sequences.shape[0]
    index = np.arange(sample_size, dtype = int)
    
    lightCurve_sequences = lightCurve_sequences[index]
    lightCurve_labels = lightCurve_labels[index]
    
    X_last_obsv = X_last_obsv[index]
    M = M[index]
    δ = δ[index]
    lightCurve_sequences = np.expand_dims(lightCurve_sequences, axis=1)
    X_last_obsv = np.expand_dims(X_last_obsv, axis=1)
    M = np.expand_dims(M, axis=1)
    δ = np.expand_dims(δ, axis=1)
    dataset_agger = np.concatenate((lightCurve_sequences, X_last_obsv, M, δ), axis = 1)
        
    train_index = int(np.floor(sample_size * train_propotion))
    valid_index = int(np.floor(sample_size * ( train_propotion + valid_propotion)))
    
    train_data, train_label = dataset_agger[:train_index], lightCurve_labels[:train_index]
    valid_data, valid_label = dataset_agger[train_index:valid_index], lightCurve_labels[train_index:valid_index]
    test_data, test_label = dataset_agger[valid_index:], lightCurve_labels[valid_index:]
    
    train_data, train_label = torch.Tensor(train_data), torch.Tensor(train_label)
    valid_data, valid_label = torch.Tensor(valid_data), torch.Tensor(valid_label)
    test_data, test_label = torch.Tensor(test_data), torch.Tensor(test_label)
    train_dataset = utils.TensorDataset(train_data, train_label)
    valid_dataset = utils.TensorDataset(valid_data, valid_label)
    test_dataset = utils.TensorDataset(test_data, test_label)
    train_dataloader = utils.DataLoader(train_dataset, batch_size = BATCH_SIZE, shuffle=False, drop_last = True)
    valid_dataloader = utils.DataLoader(valid_dataset, batch_size = BATCH_SIZE, shuffle=False, drop_last = True)
    test_dataloader = utils.DataLoader(test_dataset, batch_size = BATCH_SIZE, shuffle=False, drop_last = True)
    
    X_mean = np.mean(lightCurve_sequences, axis = 0)
    
    print('Finished')
    
    return train_dataloader, valid_dataloader, test_dataloader, max_lightCurve, X_mean



def Train_Model(model, train_dataloader, valid_dataloader, num_epochs = 500, patience = 10, min_delta = 0.001):
    
    print('Model Structure: ', model)
    print('Start Training ... ')
    
    model.cpu()
    
    if (type(model) == nn.modules.container.Sequential):
        output_last = model[-1].output_last
        print('Output type dermined by the last layer')
    else:
        output_last = model.output_last
        print('Output type dermined by the model')
        
    loss_MSE = torch.nn.MSELoss()
    loss_L1 = torch.nn.L1Loss()
    
    learning_rate = 0.001
    optimizer = torch.optim.RMSprop(model.parameters(), lr = learning_rate, alpha=0.99)
    use_gpu = torch.cuda.is_available()
    
    interval = 1
    losses_train = []
    losses_valid = []
    losses_epochs_train = []
    losses_epochs_valid = []
    
    cur_time = time.time()
    pre_time = time.time()
    
    # Variables for Early Stopping
    is_best_model = 0
    patient_epoch = 0
    for epoch in range(num_epochs):
        
        trained_number = 0
        
        valid_dataloader_iter = iter(valid_dataloader)
        
        losses_epoch_train = []
        losses_epoch_valid = []
        
        for data in train_dataloader:
            inputs, labels = data

            if inputs.shape[0] != batch_size:
                continue

            if use_gpu:
                inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
            else: 
                inputs, labels = Variable(inputs), Variable(labels)
            
            model.zero_grad()

            outputs = model(inputs)
            
            templabel = labels[:,0,:]
            
           
            if output_last:
                loss_train = loss_MSE(outputs, labels[:,0,:])
            else:
                full_labels = torch.cat((inputs[:,1:,:], labels), dim = 1)
                loss_train = loss_MSE(outputs, full_labels)
        
            losses_train.append(loss_train.data)
            losses_epoch_train.append(loss_train.data)
            
            optimizer.zero_grad()
            
            loss_train.backward()
            
            optimizer.step()
            
             # validation 
            try: 
                inputs_val, labels_val = next(valid_dataloader_iter)
            except StopIteration:
                valid_dataloader_iter = iter(valid_dataloader)
                inputs_val, labels_val = next(valid_dataloader_iter)
            
            if use_gpu:
                inputs_val, labels_val = Variable(inputs_val.cuda()), Variable(labels_val.cuda())
            else: 
                inputs_val, labels_val = Variable(inputs_val), Variable(labels_val)
                
            model.zero_grad()
            
            outputs_val = model(inputs_val)
            
            if output_last:
                loss_valid = loss_MSE(torch.squeeze(outputs_val), torch.squeeze(labels_val[:,0,:]))
            else:
                full_labels_val = torch.cat((inputs_val[:,1:,:], labels_val), dim = 1)
                loss_valid = loss_MSE(outputs_val, full_labels_val)

            losses_valid.append(loss_valid.data)
            losses_epoch_valid.append(loss_valid.data)
            
            # output
            trained_number += 1
            
        avg_losses_epoch_train = sum(losses_epoch_train).cpu().numpy() / float(len(losses_epoch_train))
        avg_losses_epoch_valid = sum(losses_epoch_valid).cpu().numpy() / float(len(losses_epoch_valid))
        losses_epochs_train.append(avg_losses_epoch_train)
        losses_epochs_valid.append(avg_losses_epoch_valid)
        best_model = model
        
        
        # Print training parameters
        cur_time = time.time()
        print('Epoch: {}, train_loss: {}, valid_loss: {}, time: {}, best model: {}'.format( \
                    epoch, \
                    np.around(avg_losses_epoch_train, decimals=8),\
                    np.around(avg_losses_epoch_valid, decimals=8),\
                    np.around([cur_time - pre_time] , decimals=2),\
                    is_best_model) )
        pre_time = cur_time
                
    return best_model, [losses_train, losses_valid, losses_epochs_train, losses_epochs_valid]


def Test_Model(model, test_dataloader, max_lightCurve):
    
    if (type(model) == nn.modules.container.Sequential):
        output_last = model[-1].output_last
    else:
        output_last = model.output_last
    
    inputs, labels = next(iter(test_dataloader))
    [batch_size, type_size, step_size, fea_size] = inputs.size()

    cur_time = time.time()
    pre_time = time.time()
    
    use_gpu = torch.cuda.is_available()
    
    loss_MSE = torch.nn.MSELoss()
    loss_L1 = torch.nn.MSELoss()
    
    tested_batch = 0
    
    losses_mse = []
    losses_l1 = [] 
    MAEs = []
    MAPEs = []
    outputArray = []
    labelArray = []
    for data in test_dataloader:
        inputs, labels = data
        if inputs.shape[0] != batch_size:
            continue
    
        if use_gpu:
            inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
        else: 
            inputs, labels = Variable(inputs), Variable(labels)
        
        outputs = model(inputs)
        #print('output',outputs,'label',labels)
    
        loss_MSE = torch.nn.MSELoss()
        loss_L1 = torch.nn.L1Loss()
        
        if output_last:
            loss_mse = loss_MSE(torch.squeeze(outputs), torch.squeeze(labels[:,0,:]))
            loss_l1 = loss_L1(torch.squeeze(outputs), torch.squeeze(labels[:,0,:]))
            MAE = torch.mean(torch.abs(torch.squeeze(outputs) - torch.squeeze(labels[:,0,:])))
            MAPE = torch.mean(torch.abs(torch.squeeze(outputs) - torch.squeeze(labels[:,0,:])) / torch.squeeze(labels[:,0,:]))
        else:
            loss_mse = loss_MSE(outputs[:,-1,:], labels[:,0,:])
            loss_l1 = loss_L1(outputs[:,-1,:], labels[:,0,:])
            MAE = torch.mean(torch.abs(outputs[:,-1,:] - torch.squeeze(labels[:,0,:])))
            MAPE = torch.mean(torch.abs(outputs[:,-1,:] - torch.squeeze(labels[:,0,:])) / torch.squeeze(labels[:,0,:]))
            
        losses_mse.append(loss_mse.data)
        losses_l1.append(loss_l1.data)
        MAEs.append(MAE.data)
        MAPEs.append(MAPE.data)
        labels = labels.detach().numpy()
        
        labelArray.append(labels[:,0,0])
        outputs = outputs.detach().numpy()
        outputArray.append(outputs[:,0])
        tested_batch += 1
    
        if tested_batch % 1000 == 0:
            cur_time = time.time()
            print('Tested #: {}, loss_l1: {}, loss_mse: {}, time: {}'.format( \
                  tested_batch * batch_size, \
                  np.around([loss_l1.data[0]], decimals=8), \
                  np.around([loss_mse.data[0]], decimals=8), \
                  np.around([cur_time - pre_time], decimals=8) ) )
            pre_time = cur_time
    losses_l1 = np.array(losses_l1)
    losses_mse = np.array(losses_mse)
    MAEs = np.array(MAEs)
    MAPEs = np.array(MAPEs)
    
    mean_l1 = np.mean(losses_l1) * max_lightCurve
    std_l1 = np.std(losses_l1) * max_lightCurve
    MAE_ = np.mean(MAEs) * max_lightCurve
    MAPE_ = np.mean(MAPEs) * 100
    outputArray = np.asarray(outputArray)
    labelArray = np.asarray(labelArray)
   
    outputArray = outputArray.reshape(outputArray.shape[0]*outputArray.shape[1])
    outputArray = outputArray * max_lightCurve
    labelArray = labelArray.reshape(labelArray.shape[0]*labelArray.shape[1])
    labelArray = labelArray * max_lightCurve
   
    n = outputArray.shape[0]
    indices = np.arange(n)
   
    plt.plot(indices,labelArray,'-b',label='Actual')
    plt.plot(indices,outputArray,'--r', label='Predicted')
    plt.xlabel('Date')
    plt.ylabel('Magnitude')
    plt.show()  
    
    print('Tested: L1_mean: {}, L1_std: {}, MAE: {} MAPE: {}'.format(mean_l1, std_l1, MAE_, MAPE_))
    return [losses_l1, losses_mse, mean_l1, std_l1]


if __name__ == "__main__":
    
    lightCurve_matrix = read_csv('V363Lyr.csv', header=0, infer_datetime_format=True, parse_dates=['Datetime'], index_col=['Datetime'])
    lightCurve_matrix = lightCurve_matrix.fillna(0)
    
        
    train_dataloader, valid_dataloader, test_dataloader, max_lightCurve, X_mean = PrepareDataset(lightCurve_matrix, BATCH_SIZE = 64)
    
    inputs, labels = next(iter(train_dataloader))
    [batch_size, type_size, step_size, fea_size] = inputs.size()
    input_dim = fea_size
    hidden_dim = fea_size
    output_dim = fea_size
    
    grud = GRUD_RNN(input_dim, hidden_dim, output_dim, X_mean, output_last = True)
    best_grud, losses_grud = Train_Model(grud, train_dataloader, valid_dataloader)
    [losses_l1, losses_mse, mean_l1, std_l1] = Test_Model(best_grud, test_dataloader, max_lightCurve)







