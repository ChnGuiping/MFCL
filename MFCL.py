import os
import random
import time
import warnings
import argparse
import shutil
import os.path as osp

import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.optim import SGD
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
import torch.nn.functional as F

import datasets
from utils import utils
from alignment.osbp import ImageClassifier as Classifier, DomainAdversarialLoss, SupConLoss, Known_class_detection, Unknown_class_detection, get_prototype_weight, bce_loss
from modules.domain_discriminator import DomainDiscriminator
from utils.data import ForeverDataIterator
from utils.meter import AverageMeter, ProgressMeter
from utils.metric import compute_h_score


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



def main(args: argparse.Namespace):
    print(args)

    cudnn.benchmark = True

    # Data loading code
    data_loaders = getattr(datasets, args.data_name)
    train_source_loader, train_target_loader, val_source_loader, val_target_loader, num_classes = data_loaders(args.data_dir, args.src_name, args.tgt_name, args.src_classes, args.tgt_classes, args.batch_size).create_loaders()

    train_source_iter = ForeverDataIterator(train_source_loader)
    train_target_iter = ForeverDataIterator(train_target_loader)

    # create model
    print("=> using pre-trained model '{}'".format(args.arch))
    backbone = utils.get_model(args.arch)
    classifier = Classifier(backbone, num_classes, bottleneck_dim=args.bottleneck_dim).to(device)
    domain_discri = DomainDiscriminator(in_feature=classifier.features_dim, hidden_size=1024).to(device)
    print(classifier, domain_discri)

    # define loss function
    ctl_fn = SupConLoss(temperature=0.3).cuda()
    unknown_csl_fn = Unknown_class_detection(num_classes).cuda()
    known_csl_fn = Known_class_detection(num_classes).cuda()
    domain_adv = DomainAdversarialLoss(domain_discri).to(device)

    # define optimizer and lr scheduler
    parameter_list = classifier.get_parameters() + domain_discri.get_parameters() #+ aug.get_parameters()
    optimizer = SGD(parameter_list, args.lr, momentum=args.momentum, weight_decay=args.wd, nesterov=True)
    lr_scheduler = LambdaLR(optimizer, lambda x: args.lr * (1. + args.lr_gamma * float(x)) ** (-args.lr_decay))


    # start training
    start_train = time.time()
    best_h_score = 0.
    for epoch in range(args.epochs):
        # train for one epoch
        train(train_source_iter, train_target_iter, classifier, domain_adv, ctl_fn, unknown_csl_fn, known_csl_fn,
              optimizer, lr_scheduler, epoch, num_classes, args)

        # evaluate on validation set
        h_score = validate(args, classifier, val_target_loader)

        # remember best acc@1 and save checkpoint
        torch.save(classifier.state_dict(), os.path.join(os.path.join(args.log, "checkpoints"), "latest.pth"))
        if h_score > best_h_score:
            shutil.copy(os.path.join(os.path.join(args.log, "checkpoints"), "latest.pth"), os.path.join(os.path.join(args.log, "checkpoints"), "best.pth"))
        best_h_score = max(h_score, best_h_score)

    print("best_h_score = {:3.2f}".format(best_h_score))
    end_train = time.time()
    print(f"训练总耗时: {end_train - start_train:.2f} 秒")




def train(train_source_iter: ForeverDataIterator, train_target_iter: ForeverDataIterator, model: Classifier,
          domain_adv: DomainAdversarialLoss, ctl_fn: nn.Module, unknown_csl_fn: nn.Module, known_csl_fn: nn.Module,
          optimizer: SGD, lr_scheduler: LambdaLR, epoch: int, num_classes, args: argparse.Namespace):
    batch_time = AverageMeter('Time', ':3.2f')
    data_time = AverageMeter('Data', ':3.2f')
    losses = AverageMeter('Loss', ':3.2f')
    sce_losses = AverageMeter('Sce Loss', ':3.2f')
    esl_losses = AverageMeter('Esl Loss', ':3.2f')
    csl_losses = AverageMeter('Cst Loss', ':3.2f')
    prt_losses = AverageMeter('Prt Loss', ':3.2f')
    adv_losses = AverageMeter('Adv Loss', ':3.2f')
    ctl_losses = AverageMeter('Tdl Loss', ':3.2f')

    progress = ProgressMeter(args.iters_per_epoch, [losses, sce_losses, esl_losses, csl_losses, prt_losses, adv_losses, ctl_losses], prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()
    domain_adv.train()

    end = time.time()

    for i in range(args.iters_per_epoch):

        (x_s, x_s_s1, x_s_s2), labels_s = next(train_source_iter)
        (x_t_w, x_t_s1, x_t_s2), labels_t = next(train_target_iter)
        bsz = labels_s.shape[0]

        x_s = x_s.to(device)
        labels_s = labels_s.to(device)

        x_t = torch.cat([x_t_w, x_t_s1], dim=0)
        x_t = x_t.to(device)
        x_t_w = x_t_w.to(device)
        x_t_s1 = x_t_s1.to(device)
        x_t_s2 = x_t_s2.to(device)
        labels_t = labels_t.to(device)

        # measure data loading time
        data_time.update(time.time() - end)

        # compute output
        y_s, f_s = model(x_s)
        y_t, f_t = model(x_t)
        y_t_w, y_t_s1 = torch.split(y_t, [bsz, bsz], dim=0)
        f_t_w, f_t_s1 = torch.split(f_t, [bsz, bsz], dim=0)

        # compute prototype
        softmax_s = nn.Softmax(dim=1)(y_s)
        softmax_t1 = nn.Softmax(dim=1)(y_t_w)
        softmax_t2 = nn.Softmax(dim=1)(y_t_s1)

        gt_list = labels_s.tolist()
        gt_list = np.unique(gt_list)

        if i == 0:
            center_feat = torch.randn(softmax_s.shape[1], f_s.shape[1]).cuda().detach()
            target_center_feat1 = torch.randn(softmax_t1.shape[1], f_t_w.shape[1]).cuda().detach()
            target_center_feat2 = torch.randn(softmax_t2.shape[1], f_t_w.shape[1]).cuda().detach()
        else:
            for c in range(len(gt_list)):
                c_idx = (labels_s == gt_list[c]).nonzero().squeeze()
                c_feat = torch.index_select(f_s, 0, c_idx)
                c_ctr = torch.mean(c_feat, dim=0)
                center_feat[gt_list[c], :] = 0.5 * center_feat[gt_list[c], :].detach() + (
                            1 - 0.5) * c_ctr.squeeze().detach()

            out_t_t1 = F.softmax(y_t_w, dim=1)
            pseudo_mask = out_t_t1.argmax(dim=1)
            gt_list = pseudo_mask.tolist()
            gt_list = np.unique(gt_list)
            for c in range(len(gt_list)):
                c_idx = (pseudo_mask == gt_list[c]).nonzero().squeeze()
                c_feat = torch.index_select(f_t_w, 0, c_idx)
                c_ctr = torch.mean(c_feat, dim=0)
                target_center_feat1[gt_list[c], :] = 0.5 * target_center_feat1[gt_list[c], :].detach() + (
                            1 - 0.5) * c_ctr.squeeze().detach()

            out_t_t2 = F.softmax(y_t_s1, dim=1)
            pseudo_mask = out_t_t2.argmax(dim=1)
            gt_list = pseudo_mask.tolist()
            gt_list = np.unique(gt_list)
            for c in range(len(gt_list)):
                c_idx = (pseudo_mask == gt_list[c]).nonzero().squeeze()
                c_feat = torch.index_select(f_t_s1, 0, c_idx)
                c_ctr = torch.mean(c_feat, dim=0)
                target_center_feat2[gt_list[c], :] = 0.5 * target_center_feat2[gt_list[c], :].detach() + (
                            1 - 0.5) * c_ctr.squeeze().detach()

        target_center1 = target_center_feat1.detach()
        target_center2 = target_center_feat2.detach()

        weight_1, _ = get_prototype_weight(target_center1, f_t_w, num_classes)
        weight_2, _ = get_prototype_weight(target_center2, f_t_w, num_classes)
        pred_weight_1 = y_t_w * weight_1
        mask_u_w_1 = pred_weight_1.argmax(dim=1)
        pred_weight_2 = y_t_s1 * weight_2
        mask_u_w_2 = pred_weight_2.argmax(dim=1)

        if epoch > args.pretrain_epoch:
            y_t_t, _ = model(x_t_w.to(device), grad_reverse=True)

            out_t_t = F.softmax(y_t_t, dim=1)
            prob1_t = torch.sum(out_t_t[:, :num_classes - 1], 1).view(-1, 1)
            prob2_t = out_t_t[:, num_classes - 1].contiguous().view(-1, 1)
            esl_loss = bce_loss(prob1_t, prob2_t)

            num_lb, num_ulb = x_s.shape[0], x_t_w.shape[0]
            preds, preds_fp = model(torch.cat((x_s, x_t_w)), need_fp=True)

            pred_s, pred_u_w = preds.split([num_lb, num_ulb])
            pred_u_w = pred_u_w.detach()
            pred_u_w = F.softmax(pred_u_w, dim=1)

            pred_u_w_fp = preds_fp[num_lb:]
            pred_u_w_fp = F.softmax(pred_u_w_fp, dim=1)

            pred_u_s, _ = model(torch.cat((x_t_s1, x_t_s2)))
            pred_u_s1, pred_u_s2 = pred_u_s.chunk(2)
            pred_u_s1 = F.softmax(pred_u_s1, dim=1)
            pred_u_s2 = F.softmax(pred_u_s2, dim=1)

            unknown_csl = unknown_csl_fn(logits_s1=pred_u_s1, logits_s2=pred_u_s2, pred_u_w_fp=pred_u_w_fp,
                                         mask_u_1=mask_u_w_1, mask_u_2=mask_u_w_2).cuda()

            known_csl = known_csl_fn(logits_s1=pred_u_s1, logits_s2=pred_u_s2, pred_u_w_fp=pred_u_w_fp,
                                     mask_u_1=mask_u_w_1, mask_u_2=mask_u_w_2, pred_u_w=pred_u_w).cuda()

            _, prt_loss = get_prototype_weight(center_feat, f_t_w, num_classes)

            adv_loss = domain_adv(f_s, f_t_w)

        else:
            unknown_csl = torch.tensor([0]).cuda()
            known_csl = torch.tensor([0]).cuda()
            prt_loss = torch.tensor([0]).cuda()
            esl_loss = torch.tensor([0]).cuda()
            adv_loss = torch.tensor([0]).cuda()


        sce_loss = F.cross_entropy(y_s, labels_s)

        ctl_loss = ctl_fn(f_t)

        csl_loss = unknown_csl + known_csl

        loss = sce_loss + esl_loss + csl_loss + 0.5 * (prt_loss + adv_loss + ctl_loss)


        losses.update(loss.item(), labels_s.size(0))
        sce_losses.update(sce_loss.item(), labels_s.size(0))
        esl_losses.update(esl_loss.item(), labels_s.size(0))
        csl_losses.update(csl_loss.item(), labels_s.size(0))
        prt_losses.update(prt_loss.item(), labels_s.size(0))
        adv_losses.update(adv_loss.item(), labels_s.size(0))
        ctl_losses.update(ctl_loss.item(), labels_s.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)




def validate(args, model, dataloader, src_flg=False):
    model.eval()

    gt_label_stack = []
    pred_cls_stack = []

    if src_flg:
        class_list = args.src_classes
        open_flg = False
    else:
        class_list = args.src_classes + [max(args.src_classes) + 1]
        open_flg = True

    for imgs_test, imgs_label in dataloader:

        imgs_test = imgs_test.cuda()

        pred_cls = model(imgs_test)
        gt_label_stack.append(imgs_label)
        pred_cls_stack.append(pred_cls.cpu())

    gt_label_all = torch.cat(gt_label_stack, dim=0) #[N]
    pred_cls_all = torch.cat(pred_cls_stack, dim=0) #[N, C]

    h_score, all_acc, known_acc, unknown_acc = compute_h_score(args, class_list, gt_label_all, pred_cls_all, open_flg)

    print(' * All {all:.5f} Known {known:.5f} Unknown {unknown:.5f} H-score {h_score:.5f}'
          .format(all=all_acc, known=known_acc, unknown=unknown_acc, h_score=h_score))

    return h_score




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='mfcl for Openset Domain Adaptation')
    # dataset parameters
    parser.add_argument('--data_name', type=str, default='SDUST', help='the name of the data')
    parser.add_argument('--data_dir', type=str, default=r'D:\Pycharm\Datas\SDUST\轴承数据集', help='the directory of the data')
    parser.add_argument('--src_name', type=str, default=[2], help='transfer learning tasks')
    parser.add_argument('--tgt_name', type=str, default=[3], help='transfer learning tasks')
    parser.add_argument('--src_classes', type=str, default=[0, 1, 2, 3, 4, 5, 6], help='transfer learning tasks')
    parser.add_argument('--tgt_classes', type=str, default=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], help='transfer learning tasks')
    parser.add_argument('--normlizetype', type=str, default='mean-std', help='nomalization type')

    # model parameters
    parser.add_argument('-a', '--arch', metavar='ARCH', default='lenet',
                        choices=utils.get_model_names(),help='backbone architecture: ' + ' | '.join(utils.get_model_names()) +' (default: resnet18)')
    parser.add_argument('--no-pool', action='store_true',
                        help='no pool layer after the feature extractor.')
    parser.add_argument('--bottleneck-dim', default=256, type=int,
                        help='Dimension of bottleneck')
    # training parameters
    parser.add_argument('-b', '--batch-size', default=32, type=int,
                        metavar='N',
                        help='mini-batch size (default: 32)')
    parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
                        metavar='LR', help='initial learning rate', dest='lr')
    parser.add_argument('--lr-gamma', default=0.0003, type=float, help='parameter for lr scheduler')
    parser.add_argument('--lr-decay', default=0.75, type=float, help='parameter for lr scheduler')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--wd', '--weight-decay', default=0.0005, type=float,
                        metavar='W', help='weight decay (default: 5e-4)')
    parser.add_argument('-j', '--workers', default=2, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--epochs', default=50, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--pretrain-epoch', default=5, type=int, help='pretrain epoch for discriminative feature learning')
    parser.add_argument('-i', '--iters-per-epoch', default=500, type=int,
                        help='Number of iterations per epoch')
    parser.add_argument('-p', '--print-freq', default=100, type=int,
                        metavar='N', help='print frequency (default: 100)')
    parser.add_argument('--seed', default=None, type=int,
                        help='seed for initializing training. ')
    parser.add_argument('--per-class-eval', action='store_true',
                        help='whether output per-class accuracy during evaluation')
    parser.add_argument("--log", type=str, default='logs',
                        help="Where to save logs, checkpoints and debugging images.")

    args = parser.parse_args()
    main(args)
