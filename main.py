import torch, argparse
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter
from network import Discriminator, Domain_adaption_model
from getloss import DAANLoss
from utils import create_logger,load_data,set_seed,apply_layered_smote
import numpy as np
from torch.optim.optimizer import Optimizer
from typing import Optional
import math
import os
from sklearn.metrics import confusion_matrix
from scipy.stats import zscore

class StepwiseLR_GRL:
    def __init__(self, optimizer: Optimizer, init_lr: Optional[float] = 0.01,
                 gamma: Optional[float] = 0.001, decay_rate: Optional[float] = 0.75, max_iter: Optional[float] = 1000):
        self.init_lr = init_lr
        self.gamma = gamma
        self.decay_rate = decay_rate
        self.optimizer = optimizer
        self.iter_num = 0
        self.max_iter = max_iter

    def get_lr(self) -> float:
        lr = self.init_lr / (1.0 + self.gamma * (self.iter_num / self.max_iter)) ** (self.decay_rate)
        return lr

    def step(self):
        """Increase iteration number `i` by 1 and update learning rate in `optimizer`"""
        lr = self.get_lr()
        for param_group in self.optimizer.param_groups:
            if 'lr_mult' not in param_group:
                param_group['lr_mult'] = 1.
            param_group['lr'] = lr * param_group['lr_mult']

        self.iter_num += 1


def test(test_loader, model, criterion, cuda):
    model.eval()
    correct = 0
    confusion_matrixs =0
    for _, (test_input, label,test_video,test_bio) in enumerate(test_loader):
        test_input, label ,test_video,test_bio= test_input.to(args.device), label.to(args.device), test_video.to(args.device),test_bio.to(args.device)
        test_input, label ,test_video,test_bio= Variable(test_input), Variable(label), Variable(test_video), Variable(test_bio)
        output = model.target_predict(test_input,test_video,test_bio)
        loss = criterion(output, label.view(-1))
        _, pred = torch.max(output, dim=1)
        correct += pred.eq(label.data.view_as(pred)).sum()
        temp_c = confusion_matrix(label.data.squeeze().cpu(), pred.cpu())
        if temp_c.shape == (2, 2):
            confusion_matrixs += temp_c
        else:
            temp = np.zeros((2, 2))
            temp[pred.cpu()[0], pred.cpu()[0]] = temp_c
            confusion_matrixs += temp.astype(int)

    accuracy = float(correct) / len(test_loader.dataset)
    return loss, accuracy, confusion_matrixs


def getInit(train_loader, model):
    model.eval()
    for _, (tran_input, tran_indx, _ ,train_video,train_bio) in enumerate(train_loader):
        tran_input, tran_indx,train_video,train_bio = tran_input.to(args.device), tran_indx.to(args.device),train_video.to(args.device),train_bio.to(args.device)
        tran_input, tran_indx,train_video ,train_bio = Variable(tran_input), Variable(tran_indx),Variable(train_video),Variable(train_bio)
        model.get_init_banks(tran_input, tran_indx,train_video,train_bio)


class CE_Label_Smooth_Loss(nn.Module):
    def __init__(self, classes=2, epsilon=0.05, ):
        super(CE_Label_Smooth_Loss, self).__init__()
        self.classes = classes
        self.epsilon = epsilon

    def forward(self, input, target):
        log_prob = torch.nn.functional.log_softmax(input, dim=-1)
        weight = input.new_ones(input.size()) * \
                 self.epsilon / (input.size(-1) - 1.)
        weight.scatter_(-1, target.unsqueeze(-1), (1. - self.epsilon))
        loss = (-weight * log_prob).sum(dim=-1).mean()
        return loss



def main(test_id, writer, args,video_signal):
    set_seed(args.seed)
    # get data1
    target_set, source_set = get_dataset(args.dataset, test_id,video_signal)
    torch_dataset_train, souece_sample_num = apply_layered_smote(source_set, primary_modal='eeg',random_state=args.seed)
    torch_dataset_test = torch.utils.data.TensorDataset(torch.from_numpy(target_set['feature']),
                                                        torch.from_numpy(target_set['label']),torch.from_numpy(target_set['video']),torch.from_numpy(target_set['bio']))
    source_loader = torch.utils.data.DataLoader(torch_dataset_train, batch_size=args.batch_size, shuffle=True,
                                                num_workers=0, pin_memory=False)
    target_loader = torch.utils.data.DataLoader(torch_dataset_test, batch_size=args.batch_size, shuffle=True,
                                                num_workers=0, pin_memory=False)
    data_loader_dict = {"source_loader": source_loader, "target_loader": target_loader, "test_loader": target_loader}

    # Create the model
    model = Domain_adaption_model(args.in_planes, args.layers, args.hidden_1, args.hidden_2, args.cls, args.device, souece_sample_num)
    domain_discriminator = Discriminator(args.hidden_2)

    # loss criterion
    criterion = nn.CrossEntropyLoss()
    model = model.to(args.device)
    domain_discriminator = domain_discriminator.to(args.device)
    criterion = criterion.to(args.device)
    dann_loss = DAANLoss(domain_discriminator, num_class=2).to(args.device)

    # Optimizer
    optimizer = torch.optim.RMSprop(
        list(model.parameters()) + list(domain_discriminator.parameters()),
        lr=args.lr, weight_decay=args.weight_decay)
    best_acc = 0

    logger.info("----------Starting training the model----------")
    # Begin training
    getInit(data_loader_dict["source_loader"],model)
    del  target_set, torch_dataset_train, torch_dataset_test
    for epoch in range(args.epochs):
        correct = 0
        count = 0

        T = len(data_loader_dict["target_loader"].dataset) // args.batch_size
        src_examples_iter, tar_examples_iter = enumerate(data_loader_dict["source_loader"]), enumerate(data_loader_dict["target_loader"])

        for i in range(T):
            model.train()
            dann_loss.train()
            _, src_examples = next(src_examples_iter)
            _, tar_examples = next(tar_examples_iter)
            src_data, src_index, src_label_cls,src_video,src_bio = src_examples
            tar_data, _,tar_video,tar_bio = tar_examples


            src_data, src_index, src_label_cls,src_video,src_bio = src_data.to(args.device), src_index.to(args.device), src_label_cls.to(args.device).view(-1), src_video.to(args.device),src_bio.to(args.device)
            tar_data ,tar_video,tar_bio= tar_data.to(args.device), tar_video.to(args.device),tar_bio.to(args.device)

            src_data, src_index, src_label_cls,src_video,src_bio = Variable(src_data), Variable(src_index), Variable(src_label_cls), Variable(src_video),Variable(src_bio)
            tar_data,tar_video,tar_bio = Variable(tar_data), Variable(tar_video),Variable(tar_bio)

            # encoder model forward
            src_output_cls, src_feature, tar_output_cls, tar_feature, source_att, target_att, tar_label = model(src_data, tar_data,src_video,tar_video, src_bio,tar_bio,src_label_cls, src_index)
            cls_loss = criterion(src_output_cls, src_label_cls)

            tar_label = torch.argmax(tar_label, dim=1)
            target_loss = criterion(tar_output_cls, tar_label)
            global_transfer_loss = dann_loss(src_feature ,
                                          tar_feature ,
                                          src_output_cls, tar_output_cls)
            boost_factor = 2.0 * (2.0 / (1.0 + math.exp(-1 * epoch / args.epochs)) - 1)
            # update joint loss function
            optimizer.zero_grad()
            loss = cls_loss + global_transfer_loss + 2*boost_factor * (target_loss)
            loss.backward()
            optimizer.step()
            _, pred = torch.max(src_output_cls, dim=1)
            correct += pred.eq(src_label_cls.data.view_as(pred)).sum()
            count += pred.size(0)
        accuracy = float(correct) / count

        writer.add_scalar("train/loss", loss, epoch)
        writer.add_scalar("train/accuracy", accuracy, epoch)
        writer.add_scalar("train/class-loss", cls_loss, epoch)
        print("Epoch: %d, loss: %f, accuracy: %f, cls_loss: %f" % (epoch, loss, accuracy, cls_loss))

        model.eval()
        test_loss, acc, confusion_matrixs = test(data_loader_dict["test_loader"], model, criterion, args)
        if acc > best_acc:
            best_acc = acc
            best_cm = confusion_matrixs
            os.makedirs(args.output_model_dir + '/' + args.dataset, exist_ok=True)
            torch.save(model, args.output_model_dir + '/' + args.dataset + '/CrossSub_no_nsal_' + str(test_id) + '.pth')
        logger.info("Testing,  Epoch: %d,loss: %f, accuracy: %f, best accuracy: %f" % (epoch, test_loss, acc, best_acc))
        writer.add_scalar("test/loss", test_loss, epoch)
        writer.add_scalar("test/Accuracy", acc, epoch)
        writer.add_scalar("test/Best_Acc", best_acc, epoch)
        print(confusion_matrixs)
    return best_acc, source_att, target_att, confusion_matrixs,best_cm


def get_dataset(dataset, test_id,video_signal):  ## dataloading function, you should modify this function according to your environment setting.
    data, data_bio,label = load_data(dataset)
    target_feature, target_feature_bio,target_label = data[test_id], data_bio[test_id],label[test_id]
    train_idxs = list(range(args.subs))
    del train_idxs[test_id]
    source_feature, source_label,source_feature_bio = np.array(data)[train_idxs], np.array(label)[train_idxs],np.array(data_bio)[train_idxs]
    source_feature=source_feature.transpose([0,1,4,3,2]).reshape([args.subs-1,-1,4,32]).reshape([-1,4,32]).reshape([-1,128])
    source_feature_bio=source_feature_bio.reshape([-1,16]).astype(np.float32)
    source_label=np.stack([source_label]*6,axis=2).reshape([-1])
    source_feature= source_feature.astype(np.float32)
    source_label=source_label.astype(np.int64)
    source_video = np.stack([video_sinal] * (args.subs-1), axis=0).reshape([-1,10,4,1152]).astype(np.float32)

    target_feature=target_feature.transpose([0,3,2,1]).reshape([-1,4,32]).reshape([-1,128])
    target_feature_bio = target_feature_bio.reshape([-1, 16]).astype(np.float32)
    # target_feature=target_feature.reshape([target_feature.shape[0],-1])
    target_label = np.stack([target_label] * 6, axis=1).reshape([-1])
    target_feature = target_feature.astype(np.float32)
    target_label = target_label.astype(np.int64)

    target_set = {'feature': target_feature, 'label': target_label, 'video':video_signal.astype(np.float32),'bio':target_feature_bio}
    source_set = {'feature': source_feature, 'label': source_label,'video':source_video,'bio':source_feature_bio}
    return target_set, source_set

def load_video_encoder(name='deap'):
    video_signal = []
    for v in range(40):
        video_feature_path = os.path.join('../deap_video_27layers_feature', f'{v + 1}video_27layers_features.npy')
        video_feature = np.load(video_feature_path).reshape(27, 60, -1, 1152).mean(axis=2).transpose(1, 0,2)
        video_signal.append(video_feature)
    video_signal = np.array
    return video_signal


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Transfer Learning')
    parser.add_argument('--dataset', type=str, nargs='?', default='deap', help='select the dataset')
    parser.add_argument('--cls', type=int, nargs='?', default=2, help="emotion classification")
    parser.add_argument('--in_planes', type=int, nargs='?', default=[4, 32], help="the size of input plane")
    parser.add_argument('--layers', type=int, nargs='?', default=1, help="DIAM squeeze ratio")
    parser.add_argument('--hidden_1', type=int, nargs='?', default=256, help="the size of hidden 1")
    parser.add_argument('--hidden_2', type=int, nargs='?', default=384, help="the size of hidden 2")
    parser.add_argument('--k', type=int, nargs='?', default=7, help="the size of k")

    parser.add_argument('--batch_size', type=int, nargs='?', default='60', help="batch_size")
    parser.add_argument('--epochs', type=int, nargs='?', default='1000', help="epochs")
    parser.add_argument('--lr', type=float, nargs='?', default='0.001', help="learning rate")
    parser.add_argument('--weight_decay', type=float, nargs='?', default='0.0001', help="weight decay")
    parser.add_argument('--seed', type=int, nargs='?', default='20', help="random seed")
    parser.add_argument('--subs', type=int, nargs='?', default='32', help="subs")
    parser.add_argument('--device', type=str, default=torch.device("cuda:2" if torch.cuda.is_available() else "cpu"),
                        help='cuda or not')
    parser.add_argument('--layer1', type=int,  nargs='?', default='16', help="layer1")
    parser.add_argument('--layer2', type=int, nargs='?', default='20', help="layer2")

    parser.add_argument('--output_log_dir', default='./train_log', type=str,
                        help='output path, subdir under output_root')
    parser.add_argument('--output_model_dir', default='./model', type=str,
                        help='output path, subdir under output_root')
    args = parser.parse_args()
    logger = create_logger(args)
    logger.info(args)
    torch.set_num_threads(8)

    sub_acc_max = []
    source_adj = []
    target_att =[]
    sub_best_confusion_matrixs=[]

    video_sinal=load_video_encoder(args.dataset)[:,:,args.layer1:args.layer2,:]
    for i in range(40):
        video_sinal[i,:,:,:] = zscore(video_sinal[i,:,:,:])
    video_sinal=video_sinal.reshape([40,6,10,4,1152],order='C').reshape([-1,10,4,1152])
    logger.info("----------Starting training the model----------")
    for test_id in range(32):
        source_id = [i for i in range(32)]
        source_id.remove(test_id)
        logger.info("The source domain: {} \nthe target domain: {}".format(source_id, test_id))
        writer = SummaryWriter("data/tensorboard/experiment_"+str(args.dataset))
        best_acc, sub_source_att, sub_target_att,confusion_matrixs,best_cm = main(test_id, writer, args,video_sinal)
        writer.close()

