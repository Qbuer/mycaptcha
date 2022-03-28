import pytorch_lightning as pl
import torch.nn as nn
import torch
import torch.nn.functional as F
# from torch.optim.radam import RAdam
from torch import optim
from lib.scheduler import GradualWarmupScheduler
from torchvision.models import ResNet
from torchvision.models.resnet import BasicBlock
from lib.radam import RAdam
from collections.abc import Iterable
from dataModule import alphabet

from dataModule import LabeltoStr
from ctcdecode import CTCBeamDecoder

class LabelConverter(object):
    """Convert between str and label.

    NOTE:
        Insert `blank` to the alphabet for CTC.

    Args:
        alphabet (str): set of the possible characters.
        ignore_case (bool, default=True): whether or not to
            ignore all of the case.
    """

    def __init__(self, alphabet, ignore_case=True, blank=62):
        self._ignore_case = ignore_case
        if self._ignore_case:
            alphabet = alphabet.lower()
        # self.alphabet = alphabet + '-'  # for `-1` index
        self.alphabet = alphabet

        self.blank = blank

        self.dict = {}
        for i, char in enumerate(alphabet):
            # NOTE: 0 is reserved for 'blank' required by CTCLoss
            self.dict[char] = i 

    def encode(self, labels):
        """Support batch or single str.

        Args:
            labels (str or list of str): labels to convert.

        Returns:
            torch.IntTensor [length_0 + length_1 + ... length_{n-1}]: encoded labels.
            torch.IntTensor [n]: length of each labels.
        """
        if isinstance(labels, str):
            labels = [self.dict[char.lower() if self._ignore_case else char] for char in labels]
            length = [len(labels)]
        elif isinstance(labels, Iterable):
            length = [len(s) for s in labels]
            labels = ''.join(labels)
            labels, _ = self.encode(labels)
        return (torch.IntTensor(labels), torch.IntTensor(length))

    def decode(self, probs, length, raw=False, strings=True):
        """Decode encoded labels back into strings.

        Args:
            torch.IntTensor [length_0 + length_1 + ...
                length_{n - 1}]: encoded labels.
            torch.IntTensor [n]: length of each labels.

        Raises:
            AssertionError: when the labels and its length does not match.

        Returns:
            labels (str or list of str): labels to convert.
        """
        if length.numel() == 1:
            length = length[0]
            assert probs.numel() == length
            if raw:
                if strings:
                    return u''.join([self.alphabet[i - 1] for i in probs]).encode('utf-8')
                return probs.tolist()
            else:
                probs_non_blank = []
                for i in range(length):
                    # removing repeated characters and blank.
                    if (probs[i] != self.blank and (not (i > 0 and probs[i - 1] == probs[i]))):
                        if strings:
                            probs_non_blank.append(self.alphabet[probs[i]])
                        else:
                            probs_non_blank.append(probs[i].item())
                if strings:
                    return ''.join(probs_non_blank)
                return probs_non_blank
        else:
            # batch mode
            assert probs.numel() == length.sum()
            labels = []
            index = 0
            for i in range(length.numel()):
                idx_end = length[i]
                labels.append(self.decode(probs[index:index + idx_end],
                              torch.IntTensor([idx_end]), raw=raw, strings=strings))
                index += idx_end
            return labels

    def best_path_decode(self, probs, raw=False, strings=True):
        lengths = torch.full((probs.shape[1],), probs.shape[0], dtype=torch.int32)
        _, probs = probs.max(2)
        probs = probs.transpose(1, 0).contiguous().reshape(-1)
        preds = self.decode(probs, lengths, raw=raw, strings=strings)
        return preds


class CrnnModel(pl.LightningModule):
    def __init__(self, class_num=62):
        super(CrnnModel, self).__init__()
        model_ft = ResNet(BasicBlock, [2, 2, 2, 2])
        self.base_model = nn.Sequential(*list(model_ft.children())[:-2])

        self.fc1 = nn.Sequential(nn.Linear(32, 32),
                                 nn.ReLU())
        
        self.gru1a = nn.GRU(32, 64)
        self.gru1b = nn.GRU(32, 64)
        
        self.gru2a = nn.GRU(64, 128)
        self.gru2b = nn.GRU(64, 128)


        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.maxpool = nn.AdaptiveMaxPool2d(1)

        self.fc2 = nn.Sequential(nn.Dropout(0.5),
                                 nn.Linear(256, 63))

        self.criterion = nn.CTCLoss(blank=62)

        # self.decoder = LabelConverter("".join(alphabet), ignore_case=False)
        self.decoder = CTCBeamDecoder("".join(alphabet)+"_", blank_id=62, cutoff_top_n=40, log_probs_input=True)

    def forward(self, x):
        
        x = self.base_model(x)

        x = torch.reshape(self.maxpool(x), (-1, 16, 32))

        x = self.fc1(x)

        x1, _ = self.gru1a(x)
        x2, _ = self.gru1b(x)

        x = x1 + x2

        x3, _ = self.gru2a(x)
        x4, _ = self.gru2b(x)
        x = torch.concat((x3, x4), dim=2)

        x = self.fc2(x)
        
        x = nn.functional.log_softmax(x, dim=2)

        x = torch.permute(x, (1, 0, 2))

        # shape of x [seq_length, batch_size, class_num]

        return x

    def training_step(self, batch, batch_idx):
        
        x, label, _ = batch
        batch_size = x.shape[0]
        y_hat = self(x)
        input_lengths = torch.full(size=(batch_size,), fill_value=16, dtype=torch.long)
        target_lengths = torch.full(size=(batch_size,), fill_value=4, dtype=torch.long)
        loss = self.criterion(y_hat, label, input_lengths, target_lengths)
        self.log("train_loss", loss, on_epoch=True, on_step=True)
        return loss

    def on_train_epoch_end(self):
        # self.log("lr", self.lr_schedulers().get_last_lr()[0], on_epoch=True, on_step=False, prog_bar=True)
        pass

    def validation_step(self, batch, batch_idx):

        x, label, golds = batch
        batch_size = x.shape[0]
        y_hat = self(x)
        input_lengths = torch.full(size=(batch_size,), fill_value=16, dtype=torch.long)
        target_lengths = torch.full(size=(batch_size,), fill_value=4, dtype=torch.long)
        loss = self.criterion(y_hat, label, input_lengths, target_lengths)
        
        # max? search 

        # preds = self.decoder.best_path_decode(y_hat)

        # end of max search
        
        # beam search 
        y_hat = torch.permute(y_hat, (1, 0, 2))

        preds = []
        beam_results, _, _, outlens = self.decoder.decode(y_hat)

        beam_results = beam_results.tolist()
        outlens = outlens.tolist()

        for beam_result, outlen in zip(beam_results, outlens):
            preds.append(LabeltoStr(beam_result[0][:outlen[0]]))

        # end of beam search
        correct_num = 0 
        for gold, pred in zip(golds, preds):
            if gold == pred:
                correct_num += 1

        self.log("val_loss", loss, prog_bar=True, on_epoch=True, on_step=False)
        self.log('val_acc', correct_num/ batch_size,
                 on_step=False, on_epoch=True, prog_bar=True)

    def predict_step(self, batch, batch_idx):
        x, _, label = batch 
        y_hat = self(x)
        y_hat = torch.permute(y_hat, (1, 0, 2))

        preds = []
        beam_results, _, _, outlens = self.decoder.decode(y_hat)

        beam_results = beam_results.tolist()
        outlens = outlens.tolist()

        for beam_result, outlen in zip(beam_results, outlens):
            preds.append(LabeltoStr(beam_result[0][:outlen[0]]))
        return label[0], preds[0]



    def configure_optimizers(self):
        optimizer = RAdam(self.parameters(),
                      lr=3e-4,
                      betas=(0.9, 0.999),
                      weight_decay=6.5e-4)
        # optimizer = optim.Adam(self.parameters(), lr=3e-4)
        scheduler_after = optim.lr_scheduler.StepLR(optimizer,
                                                step_size=20,
                                                gamma=0.5)
        scheduler = GradualWarmupScheduler(optimizer, 8, 10, after_scheduler=scheduler_after)

        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler
        }
