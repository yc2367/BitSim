"""
DNN (Sparse) Post Training Quantiztion on ImageNet-1K dataset
"""

import sys
import os
import logging
import argparse
import torch

from src.utils.utils import str2bool, load_checkpoint, save_checkpoint
from src.utils.get_data import get_ptq_dataloader
from src.trainer.base import Trainer
from src.trainer.pruning import STrainer
from src.trainer.ptq import PTQ
from src.module.convert import convert_model

from torchvision.models import resnet18, resnet50, ResNet18_Weights, ResNet50_Weights

TRAINERS = {
    "base": Trainer,
    "sparse": STrainer,
    "ptq": PTQ,
}

parser = argparse.ArgumentParser(description='T2C Training')
parser.add_argument('--model', type=str, help='model architecture')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--lr_sch', type=str, default='step', help='learning rate scheduler')
parser.add_argument('--schedule', type=int, nargs='+', default=[60, 120], help='Decrease learning rate at these epochs.')
parser.add_argument('--momentum', type=float, default=0.9, help='Momentum.')
parser.add_argument('--epochs', type=int, default=200, help='Number of epochs to train.')
parser.add_argument('--batch_size', default=128, type=int, metavar='N', help='mini-batch size (default: 64)')
parser.add_argument('--weight-decay', default=1e-5, type=float, metavar='W', help='weight decay (default: 1e-4)', dest='weight_decay')
parser.add_argument('--log_file', type=str, default=None, help='path to log file')

# loss and gradient
parser.add_argument('--loss_type', type=str, default='cross_entropy', help='loss func')
parser.add_argument('--optimizer', type=str, default='sgd', help='optimizer')
parser.add_argument('--workers', type=int, default=16,help='number of data loading workers (default: 2)')

# dataset
parser.add_argument('--dataset', type=str, default='cifar10', help='dataset: CIFAR10 / ImageNet_1k')
parser.add_argument('--train_dir', type=str, default='./data/', help='training data directory')
parser.add_argument('--val_dir', type=str, default='./data/', help='test/validation data directory')


# model saving
parser.add_argument('--save_path', type=str, default='./save/', help='Folder to save checkpoints and log.')
parser.add_argument('--evaluate', action='store_true', help='evaluate the model')
parser.add_argument('--save_param', action='store_true', help='save the model parameters')

# Fine-tuning
parser.add_argument('--fine_tune', dest='fine_tune', action='store_true',
                    help='fine tuning from the pre-trained model, force the start epoch be zero')
parser.add_argument('--resume', default='', type=str, help='path of the pretrained model')

# amp training
parser.add_argument("--mixed_prec", type=str2bool, nargs='?', const=True, default=False, help="enable amp")

# trainer
parser.add_argument('--trainer', type=str, default='base', help='trainer type')

# ptq
parser.add_argument('--wbit', type=int, default=8, help="Weight Precision")
parser.add_argument('--abit', type=int, default=8, help="Input Precision")
parser.add_argument('--wqtype', type=str, default="adaround", help='Weight quantizer')
parser.add_argument('--xqtype', type=str, default="lsq", help='Input quantizer')
parser.add_argument('--num_samples', type=int, default=1024, help="Number of samples for calibration")

args = parser.parse_args()

def main():
    if not os.path.isdir(args.save_path):
        os.makedirs(args.save_path)

    # initialize terminal logger
    logger = logging.getLogger('training')
    if args.log_file is not None:
        fileHandler = logging.FileHandler(args.save_path+args.log_file)
        fileHandler.setLevel(0)
        logger.addHandler(fileHandler)
    streamHandler = logging.StreamHandler()
    streamHandler.setLevel(0)
    logger.addHandler(streamHandler)
    logger.root.setLevel(0)
    logger.info(args)

    trainloader, testloader, num_classes = get_ptq_dataloader(args)

    # model
    if args.model == "resnet18":
        model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    elif args.model == "resnet50":
        model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
    else:
        raise NotImplementedError(f"Unknown model architecture: {args.model}")

    # convert the model to the compression-ready model
    model = convert_model(model, wbit=32, abit=32)
    logger.info(model)

    # define the trainer
    trainer = TRAINERS[args.trainer](
        model=model,
        loss_type=args.loss_type,
        trainloader=trainloader,
        validloader=testloader,
        args=args,
        logger=logger
    )

    trainer.valid_epoch()
    logger.info("[Before PTQ] Test accuracy = {:.3f}".format(trainer.logger_dict["valid_top1"]))

    # start ptq
    trainer.fit()

    trainer.valid_epoch()
    logger.info("[W{}A{}] Test accuracy = {:.3f}".format(args.wbit, args.abit, trainer.logger_dict["valid_top1"]))

    state = {
        'state_dict': trainer.model.state_dict(),
        'acc': trainer.logger_dict["valid_top1"],
    }

    filename=f"checkpoint.pth.tar"
    save_checkpoint(state, True, args.save_path, filename=filename)
    logger.info("Model Saved")

if __name__ == '__main__':
    main()
