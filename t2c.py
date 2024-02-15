"""
T2C Example on the ImagetNet-1K dataset
"""

import os
import sys
import logging
import argparse

from src.utils.utils import str2bool, load_checkpoint
from src.utils.get_data import get_ptq_dataloader
from src.trainer.base import Trainer
from src.trainer.ptq import assign_quantizer
from src.module.convert import convert_model
from src.t2c.t2c import T2C
from src.trainer.ptq import PTQ, PTQAttention
from src.t2c.convert import Vanilla4Compress
from src.d2c.base import D2C

from torchvision.models import resnet18, resnet50, ResNet18_Weights, ResNet50_Weights

TRAINERS = {
    "base": Trainer,
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
parser.add_argument('--data_path', type=str, default='./data/', help='data directory')
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

# prune
parser.add_argument('--pruner', type=str, default='element', help='trainer type')
parser.add_argument('--prune_ratio', default=0.9, type=float, help='target prune ratio')
parser.add_argument('--drate', default=0.5, type=float, help='additional pruning rate before regrow')
parser.add_argument('--warmup', type=int, default=1, help='Number of epochs to train.')
parser.add_argument('--prune_freq', type=int, default=1000, help='Iteration gap between sparsity update')
parser.add_argument('--final_epoch', type=int, default=160, help='Final pruning epoch')

# ptq
parser.add_argument('--wbit', type=int, default=8, help="Weight Precision")
parser.add_argument('--abit', type=int, default=8, help="Input Precision")
parser.add_argument('--wqtype', type=str, default="adaround", help='Weight quantizer')
parser.add_argument('--xqtype', type=str, default="lsq", help='Input quantizer')
parser.add_argument('--num_samples', type=int, default=1024, help="Number of samples for calibration")

# bit manipulation
parser.add_argument('--N', type=int, default=4, help="Number of pruned columns")
parser.add_argument('--grp_size', type=int, default=16, help="Group size")
parser.add_argument('--flag', type=int, default=0, help="0 for signed magnitude, 1 for 2s complement")

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

    args.mixup_active = False
    trainloader, testloader, num_classes = get_ptq_dataloader(args)

    # model
    if args.model == "resnet18":
        model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    elif args.model == "resnet50":
        model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
    else:
        raise NotImplementedError(f"Unknown model architecture: {args.model}")

    # convert the model to the compression-ready model
    if args.wbit < 32 or args.wbit < 32:
        converter = Vanilla4Compress(model, wbit=args.wbit, abit=args.abit)
        model = converter.reload(wqtype=args.wqtype, xqtype=args.xqtype)
    
    # resume from the checkpoint
    if args.fine_tune:
        logger.info("=> loading checkpoint...")
        state_tmp, load_acc = load_checkpoint(ckpt=args.resume, state=model.state_dict())
        model.load_state_dict(state_tmp)
        logger.info(f"Loaded checkpoint from: {args.resume}")

    # define the trainer
    trainer = TRAINERS[args.trainer](
        model=model,
        loss_type=args.loss_type,
        trainloader=trainloader,
        validloader=testloader,
        args=args,
        logger=logger,
    )

    if args.evaluate:
        # pre-trained baseline
        trainer.valid_epoch()
        logger.info("[Pre-trained Model]: Test accuracy = {:.3f}".format(trainer.logger_dict["valid_top1"]))

        # t2c and model fuse
        t2c = T2C(model=model, swl=16, sfl=12, args=args)
        qmodel = t2c.fused_model()

        # update model
        setattr(trainer, "model", qmodel.cuda())
        trainer.valid_epoch()
        logger.info("[After fusing]: Test accuracy = {:.3f}".format(trainer.logger_dict["valid_top1"]))

        # model export
        qnn = t2c.nn2chip()
    
        setattr(trainer, "model", qnn.cuda())
        trainer.valid_epoch()
        logger.info(qnn)
        logger.info("\n[Export model]: Test accuracy = {:.3f}".format(trainer.logger_dict["valid_top1"]))

        d2c = D2C(qnn, wbit=args.wbit, args=args)
        cmodel = d2c.fit()

        setattr(trainer, "model", cmodel.cuda())
        trainer.valid_epoch()

        logger.info(f"\npruned_column_num={args.N} | func = {args.flag} | grp_size = {args.grp_size}")
        logger.info("\n[Converted model]: Test accuracy = {:.3f}".format(trainer.logger_dict["valid_top1"]))
    
    else:
        exit()

if __name__ == '__main__':
    main()