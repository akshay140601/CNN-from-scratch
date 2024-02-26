#import packages
#feel free to import more if you need
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets
import numpy as np
import torch.optim as optim
import matplotlib.pyplot as plt


#evaluate the benign accuracy of a model
def test(model, x,y,batch_size):
    model.eval()
    total=x.shape[0]
    batches=np.ceil(total/batch_size).astype(int)
    success=0
    loss=0
    for i in range(batches):
        start_index=i*batch_size
        end_index=np.minimum((i+1)*batch_size,total)
        x_batch=torch.tensor(x[start_index:end_index]).float()
        y_batch=torch.tensor(y[start_index:end_index]).long()
        output=model(x_batch)
        pred=torch.argmax(output,dim=1)
        loss+=F.cross_entropy(output,y_batch).item()
        success+=(pred==y_batch).sum().item()
    #print ("accuracy: "+str(success/total))
    accuracy = success/total
    return accuracy


#define model architecture
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x=self.conv1(x)
        x=F.max_pool2d(x, 2)
        x = F.relu(x)
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return x


#untargeted attack
#you may add parameters as you wish
# Reference (Recitation 3 Code)
'''def untargeted_attack(x, y, models, min, max):
	#TODO

    model_1, model_2 = models
    for i in range(len(x)):
        img_tensor = torch.tensor(x[i]).float()
        label = torch.tensor(y[i]).long()
        for eps in range(4, 40):
            epsilon = eps/255.
            adv_tensor = img_tensor.detach().clone()
            for i in range(10):
                adv_tensor.requires_grad = True
                pred_A = model_1(adv_tensor)
                pred_B = model_2(adv_tensor)
                model_1.zero_grad()
                model_2.zero_grad()
                loss_A=nn.CrossEntropyLoss()(pred_A,label)
                loss_A.backward()
                loss_B=nn.CrossEntropyLoss()(pred_B,label)
                loss_B.backward()
                grads=adv_tensor.grad
                with torch.no_grad():
                    adv_tensor = adv_tensor - epsilon*grads.sign()
                    eta = torch.clamp(adv_tensor - img_tensor, min=-epsilon, max=epsilon)
                    adv_tensor = torch.clamp(img_tensor + eta, min=min, max=max).detach().clone()

            pred_A = model_1(adv_tensor)
            pred_B = model_2(adv_tensor)'''


def untargeted_attack(x, y, model_1, model_2, min_val, max_val):
    batch_size = 512
    num_batches = (len(x) + batch_size - 1) // batch_size
    all_accs = {}

    for eps in range(4, 44, 10):
        epsilon = eps / 255.

        adv_batch_x_all = np.zeros((len(x), x.shape[1], x.shape[2], x.shape[3]))
        print(adv_batch_x_all.shape)
        label_all = np.zeros((len(y),))

        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(x))
            batch_x = torch.tensor(x[start_idx:end_idx]).float()
            batch_y = torch.tensor(y[start_idx:end_idx]).long()
            adv_batch_x = batch_x.detach().clone()

            for _ in range(20):
                adv_batch_x.requires_grad = True
                pred_A = model_1(adv_batch_x)
                pred_B = model_2(adv_batch_x)
                model_1.zero_grad()
                model_2.zero_grad()
                loss_A = -nn.CrossEntropyLoss()(pred_A, batch_y)
                loss_A.backward()
                loss_B = -nn.CrossEntropyLoss()(pred_B, batch_y)
                loss_B.backward()
                grads = adv_batch_x.grad
                with torch.no_grad():
                    adv_batch_x = adv_batch_x - epsilon * grads.sign()
                    eta = torch.clamp(adv_batch_x - batch_x, min=-epsilon, max=epsilon)
                    adv_batch_x = torch.clamp(batch_x + eta, min=min_val, max=max_val).detach().clone()
                    #adv_batch_x = torch.round(adv_batch_x*255)/255

            '''print(adv_batch_x.numpy().shape)
            adv_batch_x_all.append(adv_batch_x.numpy())
            label_all.append(batch_y.numpy())'''

            adv_batch_x_all[start_idx:end_idx] = adv_batch_x.numpy()
            label_all[start_idx:end_idx] = batch_y.numpy()

        accuracy_A = test(model_1, adv_batch_x_all, label_all, 512)
        accuracy_B = test(model_2, adv_batch_x_all, label_all, 512)
        all_accs[eps] = [accuracy_A, accuracy_B]

    return all_accs


#targeted attack
#you may add parameters as you wish
def targeted_attack(x, y, model_1, model_2, min_val, max_val):
    #TODO
    
    alpha = 8./255
    batch_size = 512
    num_batches = (len(x) + batch_size - 1) // batch_size
    all_accs = {}

    for eps in range(4, 44, 10):
        epsilon = eps / 255.

        adv_batch_x_all = np.zeros((len(x), x.shape[1], x.shape[2], x.shape[3]))
        print(adv_batch_x_all.shape)
        label_all = np.zeros((len(y),))

        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(x))
            batch_x = torch.tensor(x[start_idx:end_idx]).float()
            batch_y = torch.tensor(y[start_idx:end_idx]).long()
            adv_batch_x = batch_x.detach().clone()
            target = torch.full((end_idx - start_idx,), 8, dtype=torch.long)
            m=torch.zeros(adv_batch_x.shape)
            v=torch.zeros(adv_batch_x.shape)

            for i in range(80):
                adv_batch_x.requires_grad = True
                pred_A = model_1(adv_batch_x)
                pred_B = model_2(adv_batch_x)
                model_1.zero_grad()
                model_2.zero_grad()
                loss_A = nn.CrossEntropyLoss()(pred_A, target)
                loss_A.backward()
                loss_B = nn.CrossEntropyLoss()(pred_B, target)
                loss_B.backward()
                grads = adv_batch_x.grad
                with torch.no_grad():
                    t=i+1
                    m=0.9*m+0.1*grads
                    v=0.999*v+0.001*grads*grads
                    mhat=m/(1.0 - 0.9**t)
                    vhat=v/(1.0 - 0.999**t)
                    grads=mhat / (torch.sqrt(vhat) + 1e-8)
                    adv_batch_x = adv_batch_x - alpha * grads.sign()
                    eta = torch.clamp(adv_batch_x - batch_x, min=-epsilon, max=epsilon)
                    adv_batch_x = torch.clamp(batch_x + eta, min=min_val, max=max_val).detach().clone()
                    #adv_batch_x = torch.round(adv_batch_x*255)/255

            '''print(adv_batch_x.numpy().shape)
            adv_batch_x_all.append(adv_batch_x.numpy())
            label_all.append(batch_y.numpy())'''

            adv_batch_x_all[start_idx:end_idx] = adv_batch_x.numpy()
            label_all[start_idx:end_idx] = batch_y.numpy()

        accuracy_A = test(model_1, adv_batch_x_all, label_all, 512)
        accuracy_B = test(model_2, adv_batch_x_all, label_all, 512)
        all_accs[eps] = [accuracy_A, accuracy_B]

    return all_accs
         
#improved targeted attack 
#you may add parameters as you wish
def targeted_attack_improved():
    #TODO

    pass

#evaluate performance of attacks
	#TODO


if __name__ == '__main__':
    #load MNIST
    dataset_train = datasets.MNIST('../data', train=True, download=True)
    dataset_test = datasets.MNIST('../data', train=False, download=True)

    # reshape MNIST
    x_train=dataset_train.data.numpy()
    y_train=dataset_train.targets.numpy()
    x_test=dataset_test.data.numpy()
    y_test=dataset_test.targets.numpy()
    x_train=np.reshape(x_train,(60000,28,28,1))
    x_test=np.reshape(x_test,(10000,28,28,1))
    x_train=np.swapaxes(x_train, 1, 3)
    x_test=np.swapaxes(x_test, 1, 3)


    #REMINDER: the range of inputs is different from what we used in the recitation
    print (x_test.min(),x_test.max())

    modelA=Net()
    modelA.load_state_dict(torch.load("modelA.zip"))
    #accuracy_A_no_attack = test(modelA,x_test,y_test,512)
    modelB=Net()
    modelB.load_state_dict(torch.load("modelB.zip"))
    #accuracy_B_no_attack = test(modelB,x_test,y_test,512)

    # Task - 1 - Untargeted attack

    '''all_accs_task_1 = untargeted_attack(x_test, y_test, modelA, modelB, x_test.min(), x_test.max())
    print(all_accs_task_1)'''

    # Task - 2 - Targeted attack

    '''test_new = dataset_test
    test_1s = [data for data in test_new if data[1] == 1]
    x_test_1s = test_1s.data.numpy()
    x_test_1s = np.reshape(x_test_1s, (len(test_1s, 28, 28, 1)))
    x_test_1s = np.swapaxes(x_test_1s, 1, 3)
    y_test_1s = test_1s.targets.numpy()'''
    indices_label_1 = np.where(y_test == 1)[0]
    x_test_1s = x_test[indices_label_1]
    y_test_1s = y_test[indices_label_1]
    all_accs_task_2 = targeted_attack(x_test_1s, y_test_1s, modelA, modelB, x_test_1s.min(), x_test_1s.max())
    print(all_accs_task_2)
