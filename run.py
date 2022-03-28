import imp
import pytorch_lightning as pl
from pytorch_lightning import Trainer
import pytorch_lightning.callbacks as plc

# import crash_on_ipy
from dataModule import captchaDataModule
from model.res18 import Res18Model
from model.res50 import Res50Model

from model.crnn import CrnnModel

from argparse import ArgumentParser
from lib.help import load_model_path

import os


def load_callbacks():
    callbacks = []
    callbacks.append(plc.EarlyStopping(
        monitor='val_acc',
        mode='max',
        patience=50,
        min_delta=0.001
    ))

    callbacks.append(plc.ModelCheckpoint(
        monitor='val_acc',
        filename='best-{epoch:02d}-{val_acc:.3f}',
        save_top_k=1,
        mode='max',
        save_last=True
    ))

    # if args.lr_scheduler:
    #     callbacks.append(plc.LearningRateMonitor(
    #         logging_interval='epoch'))
    return callbacks

def main(args):
    pl.seed_everything(args.seed)

    data = captchaDataModule(args.data_dir, batch_size=args.batch_size, num_workers=args.num_workers)
    # model = Res18Model()
    # model = Res50Model()

    model = CrnnModel()

    args.gpus = 1
    if args.no_logger:
        args.logger = False

    if args.mode == 'train':

        args.callbacks = load_callbacks()
        if args.resume:
            assert args.resume_ver is not None, '指定一个模型'
            args.ckpt_path = load_model_path(args.resume_ver, best=False)
        
        trainer = Trainer.from_argparse_args(args)
        trainer.fit(model, data)

    elif args.mode == 'predict':

        args.logger = False
        assert args.vnum
        args.ckpt_path = load_model_path(args.vnum, best=True)
        trainer = Trainer.from_argparse_args(args)
        res = trainer.predict(model, data, ckpt_path=args.ckpt_path)

        with open(f'lightning_logs/version_{args.vnum}/b_submission.csv', 'w') as writer:
            writer.write("num,tag\n")
            for idx, tag in res:
                writer.write(f"{idx},{tag}\n")

    elif args.mode == 'val':
        args.logger = False
        assert args.vnum
        args.ckpt_path = load_model_path(args.vnum, best=True)
        trainer = Trainer.from_argparse_args(args)
        trainer.validate(model, data, ckpt_path=args.ckpt_path)
        
        os.system('rm -rf debug/*')
        for pred, gold in model.res:
            os.system(f'cp sdata/dataset1/python/val/{gold}.png debug/{gold}-{pred}.png')

        


if __name__ == '__main__':
    parser = ArgumentParser()
    # Basic Training Control
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--seed', default=4882, type=int)
    parser.add_argument('--lr', default=1e-3, type=float)

    parser.add_argument('--mode', default='train', choices=['train', 'predict', 'val'])

    # LR Scheduler
    parser.add_argument('--lr_scheduler', choices=['step', 'cosine'], type=str)
    parser.add_argument('--lr_decay_steps', default=20, type=int)
    parser.add_argument('--lr_decay_rate', default=0.5, type=float)
    parser.add_argument('--lr_decay_min_lr', default=1e-5, type=float)

    

    # Restart Control
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--resume_ver', default=None, type=str)

    parser.add_argument('--vnum', default=None, type=str)


    # Training Info
    parser.add_argument('--dataset', default='standard_data', type=str)
    parser.add_argument('--data_dir', default='data/', type=str)
    parser.add_argument('--model_name', default='standard_net', type=str)
    parser.add_argument('--loss', default='bce', type=str)
    parser.add_argument('--weight_decay', default=1e-5, type=float)
    parser.add_argument('--no_augment', action='store_true')
    parser.add_argument('--log_dir', default='lightning_logs', type=str)

    parser.add_argument('--no_logger', action='store_true')
    
    # Model Hyperparameters
    # parser.add_argument('--hid', default=64, type=int)
    # parser.add_argument('--block_num', default=8, type=int)
    # parser.add_argument('--in_channel', default=3, type=int)
    # parser.add_argument('--layer_num', default=5, type=int)

    # Other
    parser.add_argument('--aug_prob', default=0.5, type=float)

    # parser = Trainer.add_argparse_args(
    #     parser.add_argument_group(title="pl.Trainer args"))

    # Reset Some Default Trainer Arguments' Default Values
    parser.set_defaults(max_epochs=1500)

    args = parser.parse_args()

    # List Arguments
    args.mean_sen = [0.485, 0.456, 0.406]
    args.std_sen = [0.229, 0.224, 0.225]

    main(args)
