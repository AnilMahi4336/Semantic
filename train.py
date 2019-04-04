import argparse
import random
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from dataset.CamVid import CamVid
import os
from model.build_BiSeNet import BiSeNet
import torch
from tensorboardX import SummaryWriter
import tqdm
from torch.nn import functional as F
import numpy as np
from utils import poly_lr_scheduler,compute_class_accuracies,compute_mean_iou,evaluate_segmentation
from utils import reverse_one_hot, get_label_info,get_label_info1, colour_code_segmentation, compute_global_accuracy,one_hot_it

def val(args, model, dataloader, csv_path,val_ind,epoch):
    print('start val!')
    class_names,_ = get_label_info1(csv_path)
    label_info=get_label_info(csv_path)
    with torch.no_grad():
        model.eval()
        precision_record = []
        scores_list = []
        class_scores_list = []
        precision_list = []
        recall_list = []
        f1_list = []
        iou_list = []
        avg_scores_per_epoch=[]
        avg_iou_per_epoch=[]
        for i, (data, label) in enumerate(dataloader):
            if i in val_ind and torch.cuda.is_available() and args.use_gpu:
                
                data = data.cuda()
                
                label = label.cuda()
                    
                    # get RGB predict image
                    
                    
                    
                    
                    predict = model(data).squeeze()
                    predict = reverse_one_hot(predict)
                    
                    predict = colour_code_segmentation(predict.cpu().detach().numpy(), label_info)
                        
                        
                        
                        # get RGB label image
                        label = label.squeeze()
                        #labels = one_hot_it(label.cpu().detach().numpy(), label_info)
                        #label = torch.from_numpy(labels)
                        label = reverse_one_hot(label)
                        #label = colour_code_segmentation(label.cpu().detach().numpy(), label_info)
                        # compute per pixel accuracy
                        accuracy, class_accuracies, prec, rec, f1, iou = evaluate_segmentation(pred=predict, label=label, num_classes=32)
                        scores_list.append(accuracy)
                        class_scores_list.append(class_accuracies)
                        precision_list.append(prec)
                        recall_list.append(rec)
                        f1_list.append(f1)
                            iou_list.append(iou)
            #label = colour_code_segmentation(label.cpu().detach().numpy(), label_info)
            else:
                continue
    avg_score = np.mean(scores_list)
        class_avg_scores = np.mean(class_scores_list, axis=0)
        avg_scores_per_epoch.append(avg_score)
        avg_precision = np.mean(precision_list)
        avg_recall = np.mean(recall_list)
        avg_f1 = np.mean(f1_list)
        avg_iou = np.mean(iou_list)
        avg_iou_per_epoch.append(avg_iou)
        
        print("\nAverage validation accuracy for epoch # %04d = %f"% (epoch, avg_score))
        print("Average per class validation accuracies for epoch # %04d:"% (epoch))
        for index, item in enumerate(class_avg_scores):
            print("%s = %f" % (class_names[index], item))
print("Validation precision = ", avg_precision)
print("Validation recall = ", avg_recall)
print("Validation F1 score = ", avg_f1)
print("Validation IoU score = ", avg_iou)
#print('precision per pixel for validation: %.3f' % dice)



def train(args, model, optimizer, dataloader_train, dataloader_val, csv_path):
    writer = SummaryWriter()
    step = 0
    for epoch in range(args.num_epochs):
        lr = poly_lr_scheduler(optimizer, args.learning_rate, iter=epoch, max_iter=args.num_epochs)
        model.train()
        tq = tqdm.tqdm(total=len(dataloader_train) * args.batch_size)
        tq.set_description('epoch %d, lr %f' % (epoch, lr))
        loss_record = []
        for i,(data, label) in enumerate(dataloader_train):
            if torch.cuda.is_available() and args.use_gpu:
                data = data.cuda()
                label = label.cuda()
            output, output_sup1, output_sup2 = model(data)
            loss1 = torch.nn.BCEWithLogitsLoss()(output, label)
            loss2 = torch.nn.BCEWithLogitsLoss()(output_sup1, label)
            loss3 = torch.nn.BCEWithLogitsLoss()(output_sup2, label)
            loss = loss1 + loss2 + loss3
            tq.update(args.batch_size)
            tq.set_postfix(loss='%.6f' % loss)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            step += 1
            writer.add_scalar('loss_step', loss, step)
            loss_record.append(loss.item())
        tq.close()
        loss_train_mean = np.mean(loss_record)
        writer.add_scalar('loss_epoch_train', float(loss_train_mean), epoch)
        print('loss for train : %f' % (loss_train_mean))
        if epoch % args.checkpoint_step == 0 and epoch != 0:
            if not os.path.isdir(args.save_model_path):
                os.mkdir(args.save_model_path)
            torch.save(model.module.state_dict(), os.path.join(args.save_model_path, 'epoch_{}.pth'.format(epoch)))
        if epoch % args.validation_step == 0:
            
            val_ind=random.sample(range(1, len(dataloader_val)), 20)
            val(args, model, dataloader_val, csv_path,val_ind,epoch)
#writer.add_scalar('precision_val', dice, epoch)


def main(params):
    # basic parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_epochs', type=int, default=300, help='Number of epochs to train for')
    parser.add_argument('--epoch_start_i', type=int, default=0, help='Start counting epochs from this number')
    parser.add_argument('--checkpoint_step', type=int, default=10, help='How often to save checkpoints (epochs)')
    parser.add_argument('--validation_step', type=int, default=10, help='How often to perform validation (epochs)')
    parser.add_argument('--dataset', type=str, default="CamVid", help='Dataset you are using.')
    parser.add_argument('--crop_height', type=int, default=640, help='Height of cropped/resized input image to network')
    parser.add_argument('--crop_width', type=int, default=640, help='Width of cropped/resized input image to network')
    parser.add_argument('--batch_size', type=int, default=1, help='Number of images in each batch')
    parser.add_argument('--context_path', type=str, default="resnet101", help='The context path model you are using.')
    parser.add_argument('--learning_rate', type=float, default=0.01, help='learning rate used for train')
    parser.add_argument('--data', type=str, default='/path/to/data',help='path of training data')
    parser.add_argument('--num_workers', type=int, default=4, help='num of workers')
    parser.add_argument('--num_classes', type=int, default=32, help='num of object classes (with void)')
    parser.add_argument('--cuda', type=str, default='0', help='GPU ids used for training')
    parser.add_argument('--use_gpu', type=bool, default=True, help='whether to user gpu for training')
    parser.add_argument('--pretrained_model_path', type=str, default=None, help='path to pretrained model')
    parser.add_argument('--save_model_path', type=str, default=None, help='path to save model')
    
    
    args = parser.parse_args(params)
    
    # create dataset and dataloader
    train_path = os.path.join(args.data, 'train')
    train_label_path = os.path.join(args.data, 'train_labels')
    val_path = os.path.join(args.data, 'val')
    val_label_path = os.path.join(args.data, 'val_labels')
    csv_path = os.path.join(args.data, 'class_dict.csv')
    dataset_train = CamVid(train_path, train_label_path, csv_path, scale=(args.crop_height, args.crop_width), mode='train')
    #print(dataset_train)
    dataloader_train = DataLoader(
                                  dataset_train,
                                  batch_size=args.batch_size,
                                  shuffle=True,
                                  num_workers=args.num_workers
                                  )
                                  dataset_val = CamVid(val_path, val_label_path, csv_path, scale=((args.crop_height, args.crop_width)),  mode='val')
                                  dataloader_val = DataLoader(
                                                              dataset_val,
                                                              # this has to be 1
                                                              batch_size=1,
                                                              shuffle=True,
                                                              num_workers=args.num_workers
                                                              )
                                  
                                  # build model
                                  os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda
                                  model = BiSeNet(args.num_classes, args.context_path)
                                  #print(model)
                                  if torch.cuda.is_available() and args.use_gpu:
                                      model = torch.nn.DataParallel(model).cuda()

# build optimizer
optimizer = torch.optim.RMSprop(model.parameters(), args.learning_rate)

# load pretrained model if exists
if args.pretrained_model_path is not None:
    print('load model from %s ...' % args.pretrained_model_path)
    model.module.load_state_dict(torch.load(args.pretrained_model_path))
    print('Done!')
    
    # train
    train(args, model, optimizer, dataloader_train, dataloader_val, csv_path)
#val_ind=random.sample(range(1, len(dataloader_val)), 20)
#val(args, model, dataloader_val, csv_path,val_ind,epoch=20)


if __name__ == '__main__':
    params = [
              '--num_epochs', '305',
              '--learning_rate', '0.001',
              '--data', '/content/segment/dataset/CamVid',
              '--num_workers', '4',
              '--num_classes', '32',
              '--cuda', '0',
              '--batch_size', '1',
              '--save_model_path', '/content/segment'
              ]
    main(params)

