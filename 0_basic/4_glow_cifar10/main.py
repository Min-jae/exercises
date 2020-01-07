import torch
from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms
import argparse
import time
import logging
import os
# import matplotlib.pyplot as plt

from model.glow import Glow
from model.loss import GlowLoss


def save_checkpoint(checkpoint_dir, model, optimizer, iterations, loss, loss_test):
    """FUNCTION TO SAVE CHECKPOINT
    Args:
        checkpoint_dir (str): directory to save checkpoint
        model (torch.nn.Module): pytorch model instance
        optimizer (Optimizer): pytorch optimizer instance
        iterations (int): number of current iterations
    """
    checkpoint = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "iterations": iterations,
        "loss": loss,
        "loss_test": loss_test}

    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    torch.save(checkpoint, checkpoint_dir + "/checkpoint-%d.pkl" % iterations)


def _train_loop(train_loader, model, criterion, optimizer, use_cuda):
    loss_avg = 0.
    for image, label in train_loader:
        if use_cuda:
            image, label = image.cuda(), label.cuda()
        z, logdet = model(image, logdet=0., reverse=False)
        loss = criterion(z, logdet)
        loss.backward()
        optimizer.step()
        loss_avg += loss / len(train_loader)

    return loss_avg


def _test_loop(test_loader, model, criterion, use_cuda):
    loss_avg = 0.
    for image, label in test_loader:
        if use_cuda:
            image, label = image.cuda(), label.cuda()
        z, logdet = model(image, logdet=0., reverse=False)
        loss = criterion(z, logdet)
        loss_avg += loss / len(test_loader)

    return loss_avg


def main():
    use_cuda = torch.cuda.is_available()

    args = get_arguments()
    # make experiment directory
    expdir = "./exp/exp"
    for key, value in vars(args).items():
        print("{} = {}".format(str(key), str(value)))
        if key not in "dataset_path":
            expdir += "_" + str(key) + str(value)

    if not os.path.exists(expdir):
        os.makedirs(expdir, exist_ok=True)
        os.makedirs(expdir + "/models")

    torch.save(args, expdir + "/model.conf")

    # load MNIST database
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    train_dataset = CIFAR10(args.dataset_path, transform=transform, train=True, download=True)
    test_dataset = CIFAR10(args.dataset_path, transform=transform, train=False, download=True)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=args.batch_size,
                                               shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=args.batch_size,
                                              shuffle=False)

    model = Glow(n_chn=3, n_flow=args.n_flow, squeeze_layer=args.squeeze_layer)
    criterion = GlowLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    if use_cuda:
        model.cuda()
        criterion.cuda()
        for state in optimizer.state.values():
            for key, value in state.items():
                if torch.is_tensor(value):
                    state[key] = value.cuda()

    print('Training start\n')

    model.train()
    for epoch in range(args.n_epoch):
        start_t = time.time()
        loss = _train_loop(train_loader=train_loader, model=model, criterion=criterion, optimizer=optimizer, use_cuda=use_cuda)
        loss_test = _test_loop(test_loader=test_loader, model=model, criterion=criterion, use_cuda=use_cuda)
        print('Epoch: {:} \tTrain/Test loss: {:.6f} / {:.6f}, time required: {.2f} sec'.format(
            epoch + 1, loss, loss_test, time.time() - start_t))
        save_checkpoint(expdir + "/models", model, optimizer, epoch, loss, loss_test)


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", default="./dataset",
                        type=str, help="directory of dataset")
    parser.add_argument("--n_flow", default=12,
                        type=int, help="number of flow layer")
    parser.add_argument("--squeeze_layer", default=[0, 1, 2], nargs="*",
                        type=int, help="index of layer which performs squeezing operation")

    # Training related
    parser.add_argument("--lr", default=1e-4,
                        type=float, help="learning rate")
    parser.add_argument("--batch_size", default=64,
                        type=int, help="batch size")
    parser.add_argument("--n_epoch", default=200,
                        type=int, help="number of epoch")

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    main()
