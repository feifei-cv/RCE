import random
import time
import warnings
import argparse
import shutil
import os.path as osp

import torch ##
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.optim.lr_scheduler import LambdaLR
import torch.nn.functional as F

import common.vision.datasets as datasets
import common.vision.models as models
from common.utils.data import ForeverDataIterator
from common.utils.metric import accuracy, ConfusionMatrix
from common.utils.meter import AverageMeter, ProgressMeter
from common.utils.logger import CompleteLogger
from common.utils.analysis import collect_feature, tsne, a_distance
from network import ImageClassifier
from loss import consistency_loss
from data.prepare_data_da import generate_dataloader as Dataloader
from loss import weightAnchor, reg, ent, kld
from dalib.modules.domain_discriminator import DomainDiscriminator
from dalib.adaptation.dann import DomainAdversarialLoss


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def opts():

    architecture_names = sorted(
        name for name in models.__dict__
        if name.islower() and not name.startswith("__")
        and callable(models.__dict__[name])
    )
    dataset_names = sorted(
        name for name in datasets.__dict__
        if not name.startswith("__") and callable(datasets.__dict__[name])
    )
    parser = argparse.ArgumentParser(description='UDA')
    # dataset parameters
    parser.add_argument('root', metavar='DIR', help='root path of dataset')
    parser.add_argument('-d', '--data', metavar='DATA', default='Office31',
                        help='dataset: ' + ' | '.join(dataset_names) + ' (default: Office31)')
    parser.add_argument('-s', '--source', help='source domain(s)')
    parser.add_argument('-t', '--target', help='target domain(s)')
    parser.add_argument('--transform_type', type=str,
                        default='randomcrop', help='randomcrop | randomsizedcrop | center')
    parser.add_argument('--strongaug', action='store_true', default=True,
                        help='whether use the strong augmentation (i.e., RandomAug) it is True in FixMatch and UDA')
    parser.add_argument('--mu', type=int, default=2, help='unlabeled batch size / labeled batch size')

    # model parameters
    parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet50', choices=architecture_names,
                        help='backbone architecture: ' + ' | '.join(architecture_names) + ' (default: resnet50)')
    parser.add_argument('--bottleneck-dim', default=256, type=int, help='Dimension of bottleneck')
    parser.add_argument('--temperature', default=1.8, type=float, help='parameter temperature scaling')
    parser.add_argument('--trade-off1', default=0.5, type=float,
                        help='hyper-parameter for regularization')
    parser.add_argument('--trade-off2', default=1.0, type=float,
                        help='hyper-parameter for correct loss')
    parser.add_argument('--trade-off3', default=0.5, type=float,
                        help='fixmatch loss')
    parser.add_argument('--threshold', default=0.97, type=float)

    # training parameters
    parser.add_argument('-b', '--batch-size', default=32, type=int, metavar='N',
                        help='mini-batch size (default: 32)')
    parser.add_argument('--lr', '--learning-rate', default=0.01, type=float,
                        metavar='LR', help='initial learning rate', dest='lr')
    parser.add_argument('--lr-gamma', default=0.001, type=float, help='parameter for lr scheduler')
    parser.add_argument('--lr-decay', default=0.75, type=float, help='parameter for lr scheduler')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
    parser.add_argument('--wd', '--weight-decay', default=1e-3, type=float,
                        metavar='W', help='weight decay (default: 1e-3)', dest='weight_decay')
    parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                        help='number of data loading workers (default: 8)')
    parser.add_argument('--epochs', default=20, type=int, metavar='N', help='number of total epochs to run')
    parser.add_argument('--early', default=20, type=int, metavar='N', help='number of total epochs to early stopping')
    parser.add_argument('-i', '--iters-per-epoch', default=1000, type=int, help='Number of iterations per epoch')
    parser.add_argument('-p', '--print-freq', default=100, type=int, metavar='N', help='print frequency (default: 100)')
    parser.add_argument('--seed', default=2, type=int, help='seed for initializing training. ')
    parser.add_argument('--per-class-eval', action='store_true',
                        help='whether output per-class accuracy during evaluation')
    # pretrain parameters
    parser.add_argument('--pretrain', type=str, default=None, help='pretrain checkpoints for classification model')
    parser.add_argument('--pretrain-epochs', default=0, type=int, metavar='N',
                        help='number of total epochs(pretrain) to run')
    parser.add_argument('--pretrain-lr', '--pretrain-learning-rate', default=0.001, type=float,
                        help='initial pretrain learning rate', dest='pretrain_lr')
   ## log parameters
    parser.add_argument("--log", type=str, default='RCE',
                        help="Where to save logs, checkpoints and debugging images.")
    parser.add_argument("--phase", type=str, default='train', choices=['train', 'test', 'analysis'],
                        help="When phase is 'test', only test the model."
                             "When phase is 'analysis', only analysis the model.")
    parser.add_argument('--multiprocessing_distributed', action='store_true', default=False,
                        help='Use multi-processing distributed training to launch '
                             'N processes per node, which has N GPUs. This is the '
                             'fastest way to use PyTorch for either single node or '
                             'multi node data parallel training')
    args = parser.parse_args()

    data_dir = '/disks/disk0/feifei/paper/paper3-3090/data/' ### change your own data root
    args.root = data_dir + args.root

    if args.data == 'OfficeHome':
        args.num_class = 65
    elif args.data == 'Office31':
        args.num_class = 31
    elif args.data == 'DomainNet':
        args.num_class = 345
    elif args.data == 'VisDA2017':
        args.num_class = 12
        args.category_mean = True
        args.transform_type = 'center'
        print('training with VisDA2017')
    return args

def main():

    args = opts()
    logger = CompleteLogger(args.log, args.phase)
    print(args)
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')
    cudnn.benchmark = True

    ###### data
    dataloaders = Dataloader(args)
    train_source_loader = dataloaders['source']
    train_target_loader = dataloaders['target']
    val_loader =  dataloaders['trans_test']
    test_loader = val_loader
    train_source_iter = ForeverDataIterator(train_source_loader)
    train_target_iter = ForeverDataIterator(train_target_loader)

    ##### create model
    print("=> using pre-trained model '{}'".format(args.arch))
    backbone = models.__dict__[args.arch](pretrained=True)
    classifier = ImageClassifier(backbone, args.num_class, bottleneck_dim=args.bottleneck_dim).to(device)

    ##### optimizer and lr scheduler
    optimizer = torch.optim.SGD(classifier.get_parameters(), lr=args.lr, momentum=args.momentum,
                    weight_decay=args.weight_decay, nesterov=True)
    lr_scheduler = LambdaLR(optimizer, lambda x: args.lr * (1. + args.lr_gamma * float(x)) ** (-args.lr_decay))

    #### init matrix
    global transMatrix
    transMatrix = (torch.eye(args.num_class)).to(device)

    #### resume from the best checkpoint
    if args.phase != 'train':
        checkpoint = torch.load(logger.get_checkpoint_path('best'), map_location='cpu')
        classifier.load_state_dict(checkpoint)
    #### analysis the model
    if args.phase == 'analysis':
        # extract features from both domains
        feature_extractor = nn.Sequential(classifier.backbone, classifier.bottleneck).to(device)
        source_feature = collect_feature(train_source_loader, feature_extractor, device)
        target_feature = collect_feature(train_target_loader, feature_extractor, device)
        # plot t-SNE
        tSNE_filename = osp.join(logger.visualize_directory, 'TSNE.png')
        tsne.visualize(source_feature, target_feature, tSNE_filename)
        print("Saving t-SNE to", tSNE_filename)
        # calculate A-distance, which is a measure for distribution discrepancy
        A_distance = a_distance.calculate(source_feature, target_feature, device)
        print("A-distance =", A_distance)
        return
    if args.phase == 'test':
        acc1 = validate(test_loader, classifier, args)
        print(acc1)
        return

    #### pretrain
    args.pretrain_domain_predictor = logger.get_checkpoint_path('domain_predictor')
    domain_predictor = DomainDiscriminator(in_feature=classifier.features_dim, hidden_size=1024).to(device)
    if not osp.exists(args.pretrain_domain_predictor):
        # first pretrain the classifier wish source data
        print("Pretraining the model on source domain.")
        args.pretrain_domain_predictor = logger.get_checkpoint_path('domain_predictor')
        pretrained_model = ImageClassifier(backbone, args.num_class, bottleneck_dim=args.bottleneck_dim).to(device)
        domain_predictor = DomainDiscriminator(in_feature=classifier.features_dim, hidden_size=1024).to(device)
        pretrain_optimizer = torch.optim.Adam(pretrained_model.get_parameters()+domain_predictor.get_parameters(),
                                              args.pretrain_lr)
        pretrain_lr_scheduler = LambdaLR(pretrain_optimizer,
                                         lambda x: args.pretrain_lr * (1. + args.lr_gamma * float(x)) ** (
                                             -args.lr_decay))
        domain_adv = DomainAdversarialLoss(domain_predictor).to(device)
        # start pretraining
        for epoch in range(args.pretrain_epochs):
            # pretrain for one epoch
            pretrain(train_source_iter, train_target_iter, pretrained_model, pretrain_optimizer, pretrain_lr_scheduler,
                     epoch, domain_adv, args)
            # validate to show pretrain process
            validate(val_loader, pretrained_model, args)
        torch.save(domain_predictor.state_dict(), args.pretrain_domain_predictor)
        # show pretrain result
        pretrain_acc = validate(val_loader, pretrained_model, args)
        print("pretrain_acc1 = {:3.1f}".format(pretrain_acc))
    checkpoint = torch.load(args.pretrain_domain_predictor, map_location='cpu')
    domain_predictor.load_state_dict(checkpoint)

    #### start training
    best_acc1 = 0.
    for epoch in range(min(args.epochs, args.early)):
        print("lr:", lr_scheduler.get_last_lr()[0])
        train(train_source_iter, train_target_iter, classifier, optimizer, lr_scheduler, epoch, domain_predictor, args)
        acc1 = validate(val_loader, classifier, args)
        torch.save(classifier.state_dict(), logger.get_checkpoint_path('latest'))
        if acc1 > best_acc1:
            shutil.copy(logger.get_checkpoint_path('latest'), logger.get_checkpoint_path('best'))
        best_acc1 = max(acc1, best_acc1)
    print("best_acc1 = {:3.1f}".format(best_acc1))
    classifier.load_state_dict(torch.load(logger.get_checkpoint_path('best')))
    acc1 = validate(test_loader, classifier, args)
    print("test_acc1 = {:3.1f}".format(acc1))
    logger.close()


def pretrain(train_source_iter,train_target_iter, model, optimizer, lr_scheduler, epoch, domain_adv, args):

    batch_time = AverageMeter('Time', ':3.1f')
    data_time = AverageMeter('Data', ':3.1f')
    losses = AverageMeter('Loss', ':3.2f')
    cls_accs = AverageMeter('Cls Acc', ':3.1f')
    domain_accs = AverageMeter('Domain Acc', ':3.1f')
    progress = ProgressMeter(
        args.iters_per_epoch,
        [batch_time, data_time, losses, cls_accs, domain_accs],
        prefix="Epoch: [{}]".format(epoch))
    # switch to train mode
    model.train()
    domain_adv.train()
    end = time.time()
    for i in range(args.iters_per_epoch):
        (x_s,_), labels_s = next(train_source_iter)
        (x_t, _), _ = next(train_target_iter)
        x_s = x_s.to(device)
        x_t = x_t.to(device)
        labels_s = labels_s.to(device)
        # measure data loading time
        data_time.update(time.time() - end)
        # compute output
        y_s, f_s = model(x_s)
        y_t, f_t = model(x_t)
        cls_loss = F.cross_entropy(y_s, labels_s)
        transfer_loss = domain_adv(f_s, f_t)
        args.trade_off_adv = 1
        loss = cls_loss + transfer_loss * args.trade_off_adv
        cls_acc = accuracy(y_s, labels_s)[0]
        domain_acc = domain_adv.domain_discriminator_accuracy
        losses.update(loss.item(), x_s.size(0))
        cls_accs.update(cls_acc.item(), x_s.size(0))
        domain_accs.update(domain_acc.item(), x_s.size(0))
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

def train(train_source_iter, train_target_iter, model, optimizer, lr_scheduler, epoch, domain_predictor, args):

    global transMatrix
    batch_time = AverageMeter('Time', ':3.1f')
    data_time = AverageMeter('Data', ':3.1f')
    losses = AverageMeter('Loss', ':3.2f')
    reg_losses = AverageMeter('Reg Loss', ':3.2f')
    y_t_losses_correct = AverageMeter('correct_t loss', ':3.2f')
    ssl_losses = AverageMeter('SSL Ls', ':3.2f')
    cls_accs = AverageMeter('s_Acc', ':3.1f')
    tgt_accs = AverageMeter('t_Acc', ':3.1f')

    progress = ProgressMeter(args.iters_per_epoch,
        [losses, y_t_losses_correct, ssl_losses, cls_accs, tgt_accs, reg_losses],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()
    domain_predictor.eval()
    end = time.time()

    for i in range(args.iters_per_epoch):

        (x_s,_), labels_s = next(train_source_iter)
        (x_t, x_t_u), labels_t = next(train_target_iter)

        x_s = x_s.to(device)
        x_t = x_t.to(device)
        x_t_u = x_t_u.to(device)
        labels_s = labels_s.to(device)
        labels_t = labels_t.to(device)
        # measure data loading time
        data_time.update(time.time() - end)

        # compute output
        y_s, f_s = model(x_s)
        y_t, f_t = model(x_t)
        y_t_u, f_t_u = model(x_t_u)

        # generate target pseudo-labels
        max_prob, pred_u = torch.max(F.softmax(y_t, dim=1), dim=-1)
        ssl_loss, _, = consistency_loss(y_t, y_t_u, T=1.0, p_cutoff=0.97)  ## fixmatch loss

        ### train noise classifier: closed form
        f_t_norm = f_t / (torch.norm(f_t, dim=-1).reshape(f_t.shape[0], 1))
        f_s_norm = f_s / (torch.norm(f_s, dim=-1).reshape(f_s.shape[0], 1))
        f_t_kernel = torch.clamp(f_t_norm.mm(f_t_norm.transpose(dim0=1, dim1=0)), -0.99999999, 0.99999999) # xx'
        predict_kernel = torch.clamp(f_s_norm.mm(f_t_norm.transpose(dim0=1, dim1=0)), -0.99999999, 0.99999999) # fx'
        soft_label_t = 0.999 * torch.nn.functional.one_hot(pred_u, args.num_class) + 0.001 / float(args.num_class)

        ### source noise class posterior estimate
        class_poster_s = predict_kernel.mm(
            torch.inverse(f_t_kernel + 0.001 * torch.eye(args.mu*args.batch_size).to(device))).mm(soft_label_t)
        class_poster_s = torch.clamp(class_poster_s, 0.00000001, 0.99999999)
        class_poster_s = class_poster_s / (torch.sum(class_poster_s, dim=1).reshape(class_poster_s.shape[0], 1))

        ### target noise class posterior estimate
        class_poster_t = f_t_kernel.mm(
            torch.inverse(f_t_kernel + 0.001 * torch.eye(args.mu*args.batch_size).to(device))).mm(soft_label_t)
        class_poster_t = torch.clamp(class_poster_t, 0.00000001, 0.99999999)
        class_poster_t = class_poster_t / (torch.sum(class_poster_t, dim=1).reshape(class_poster_t.shape[0], 1))

        ### get anchor points
        anchor_domain_predictor = (domain_predictor(f_s).le(0.5)).squeeze() # 0:target, 1:source
        _, pred_s_clean = F.softmax(y_s, dim=1).max(dim=1)
        anchor_acc = (pred_s_clean == labels_s) ### distill example
        anchor_mask = anchor_domain_predictor*anchor_acc
        anchor_points = labels_s[anchor_mask]
        target_test_pred_r_select = class_poster_s[anchor_mask]
        weights = weightAnchor(f_s,temperature=args.temperature)
        weight_anchor = weights[anchor_mask]
        #### estimate T
        with torch.no_grad():
            for k in range(transMatrix.size(0)):
                indice = torch.where((anchor_points == k))
                if len(indice[0]) == 0:
                    continue
                else:
                    current = target_test_pred_r_select[indice][weight_anchor[indice].max(dim=0)[1]].detach()
                    transMatrix[k] = 0.01*current + 0.99* transMatrix[k] ##更update

        ## clean to noise
        y_t_softamx = F.softmax(y_t, dim=1)
        y_t_correct = y_t_softamx.mm(transMatrix.detach())
        y_t_loss_correct = nn.KLDivLoss()(torch.log(y_t_correct), class_poster_t.detach()/args.temperature) #
        cls_loss = F.cross_entropy(y_s, labels_s)
        reg_loss = reg(y_t, args.temperature)/(args.batch_size * args.mu)
        ### reg_loss = kld(y_t, args.num_class, args.temperature) / (args.batch_size * args.mu)
        ### reg_loss = ent(y_t, args.temperature) / (args.batch_size * args.mu)
        # y_t_u_softamx = F.softmax(y_t_u, dim=1)
        # variance = torch.sum(nn.KLDivLoss(reduction='none')(torch.log(y_t_softamx), y_t_u_softamx), dim=1)
        # exp_variance = torch.exp(-variance)
        loss = cls_loss + args.trade_off1*ssl_loss + \
               args.trade_off2*y_t_loss_correct + args.trade_off3*reg_loss

        ####print
        cls_acc = accuracy(y_s, labels_s)[0]
        tgt_acc = accuracy(y_t, labels_t)[0]
        losses.update(loss.item(), x_s.size(0))
        cls_accs.update(cls_acc.item(), x_s.size(0))
        tgt_accs.update(tgt_acc.item(), x_s.size(0))
        reg_losses.update(reg_loss.item(), x_s.size(0))
        y_t_losses_correct.update(y_t_loss_correct.item(), x_s.size(0))
        ssl_losses.update(ssl_loss.item(), x_s.size(0))

        ## backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        # # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        if i % args.print_freq == 0:
            progress.display(i)
        ## error
        # err_index_t = (pred_u != labels_t)
        # matrix_confidence = matrix_confidence.to(device)
        # matrix_number = matrix_number.to(device)
        # for c in range(len(matrix_confidence)):
        #     c_index = torch.where(labels_t == c)
        #     if len(c_index[0]) == 0:
        #         continue
        #     else:
        #         coloum = pred_u[c_index]
        #         confidence = max_prob[c_index]
        #         for i, col in enumerate(coloum):
        #             matrix_confidence[c][col] += confidence[i]
        #             matrix_number[c][col] += 1

def validate(val_loader, model, args):

    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1, top5],
        prefix='Test: ')
    # switch to evaluate mode
    model.eval()
    if args.per_class_eval:
        classes = val_loader.dataset.classes
        confmat = ConfusionMatrix(len(classes))
    else:
        confmat = None
    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            images = images.to(device)
            target = target.to(device)
            # compute output
            output, _ = model(images)
            loss = F.cross_entropy(output, target)
            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            if confmat:
                confmat.update(target, output.argmax(1))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1.item(), images.size(0))
            top5.update(acc5.item(), images.size(0))
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            if i % args.print_freq == 0:
                progress.display(i)
        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'.format(top1=top1, top5=top5))
        if confmat:
            print(confmat.format(classes))
    return top1.avg


if __name__ == '__main__':
    main()



