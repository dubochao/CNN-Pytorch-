import os
import sys
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import Optimizer
writer = SummaryWriter('./runs')
grad_clip = 1.0  # clip gradients at an absolute value of
save_prefix=''
def clip_gradient(optimizer, grad_clip):
#     """
# 剪辑反向传播期间计算的梯度，以避免梯度爆炸。
#
# param optimizer：具有要剪裁的渐变的优化器
#
# ：参数梯度剪辑：剪辑值
#     """
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)
def train(train_iter, dev_iter, model, args):
    # global args
    global save_prefix
    save_dir = args.save_dir
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    filename = args.snapshot
    save_prefix = os.path.join(save_dir, filename)

    if args.snapshot:

        snapshot = os.path.join(args.save_dir, args.snapshot)
        if os.path.exists(snapshot):
            print('\nLoading model from {}...\n'.format(snapshot))
            model = torch.load(snapshot)['model']
            optimizer=torch.load(snapshot)['optimizer']
        else:
            optimizer = Optimizer.Optimizer(
                torch.optim.Adam(model.parameters(), betas=(0.9, 0.98), eps=1e-09))
    if args.cuda:
        model.cuda()
    # optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    steps = 0
    best_acc = 0
    last_step = 0
    model.train()
    for epoch in range(1, args.epochs + 1):
        for batch in train_iter:
            feature, target = batch.text, batch.label
            
            feature.t_(), target.sub_(1)
            # w.add_graph(model, (feature,))
            if args.cuda:
                feature, target = feature.cuda(), target.cuda()
            optimizer.zero_grad()
            logits = model(feature)
            loss = F.cross_entropy(logits, target)
            loss.backward()
            # Clip gradients
            clip_gradient(optimizer.optimizer, grad_clip)
            optimizer.step()
            steps += 1
            if steps % args.log_interval == 0:
                corrects = (torch.max(logits, 1)[1].view(target.size()).data == target.data).sum()
                train_acc = corrects / batch.batch_size
                sys.stdout.write(
                    '\rBatch[{}] - loss: {:.6f}  acc: {:.4f}({}/{})'.format(steps,
                                                                             loss.item(),
                                                                             train_acc,
                                                                             corrects,
                                                                             batch.batch_size))
            writer.add_scalar('Batch/train_loss', loss.item() ,optimizer.step_num)
            writer.add_scalar('Batch/learning_rate', optimizer.lr, optimizer.step_num)
            if steps % args.test_interval == 0:
                dev_acc = eval(dev_iter, model, args,optimizer)
                if dev_acc > best_acc:
                    best_acc = dev_acc
                    last_step = steps
                    if args.save_best:
                        print('Saving best model, acc: {:.4f}\n'.format(best_acc))
                        save(model, best_acc,optimizer)
                        writer.add_scalar('best/acc', best_acc, optimizer.step_num)
                elif steps - last_step >= args.early_stopping:
                        print('\nearly stop by {} steps, acc: {:.4f}'.format(args.early_stopping, best_acc))
                        raise KeyboardInterrupt
                else:

                    # print(type(model.fc.weight),type(torch.load(save_prefix)['model'].fc.weight))
                    # print(torch.load(save_prefix)['model'].fc.weight==model.fc.weight)
                    w=model.fc.weight+ torch.load(save_prefix)['model'].fc.weight
                    # print('1')
                    b=model.fc.bias+ torch.load(save_prefix)['model'].fc.bias
                    model.fc.weight=torch.nn.Parameter(w/2)
                    model.fc.bias = torch.nn.Parameter(b / 2)
def eval(data_iter, model, args,optimizer):
    model.eval()
    corrects, avg_loss = 0, 0
    for batch in data_iter:
        feature, target = batch.text, batch.label
        feature.t_(), target.sub_(1)
        if args.cuda:
            feature, target = feature.cuda(), target.cuda()
        logits = model(feature)
        loss = F.cross_entropy(logits, target)
        avg_loss += loss.item()
        corrects += (torch.max(logits, 1)
                     [1].view(target.size()).data == target.data).sum()
    size = len(data_iter.dataset)
    avg_loss /= size
    accuracy =  corrects / size
    print('\nEvaluation - loss: {:.6f}  acc: {:.4f}({}/{}) \n'.format(avg_loss,
                                                                       accuracy,
                                                                       corrects,
                                                                       size))
    writer.add_scalar('Evaluation/train_loss', avg_loss, optimizer.step_num)
    writer.add_scalar('Evaluation/learning_rate', optimizer.lr, optimizer.step_num)
    return accuracy

def save(model, best_acc,optimizer):

    state = {
             'best_acc': best_acc,
             'model': model,
    'optimizer':optimizer}

    torch.save(state, save_prefix)
