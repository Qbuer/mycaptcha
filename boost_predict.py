# %%
from model.res18 import Res18Model
from model.res50 import Res50Model
from model.crnn import CrnnModel
from dataModule import captchaDataModule, LabeltoStr
import torch
import xgboost as xgb
from lib.help import load_model_path

from tqdm import tqdm

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

# %%
versions = [
    '8', '9', '12', '13', '14', '60', '62', '63'
]
model_type = [
    Res18Model, Res18Model, Res18Model, Res18Model, Res50Model, Res18Model, Res18Model, Res18Model
]


# %%
models = []
for v, M in zip(versions, model_type):
    path = load_model_path(v)
    models.append(M.load_from_checkpoint(path).eval())


# %%
data = captchaDataModule('data/dataset1', batch_size=16, num_workers=4)
data.setup()

# %%
dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
models = [model.to(dev) for model in models]
print(dev)

xgb_model = xgb.XGBClassifier()
xgb_model.load_model('xgb.model')

# %%
outputs = []
Y = []
tags = []
with torch.no_grad():
    for x, label, tag in tqdm(data.predict_dataloader()):
        
        x = x.to(dev)
        Y.append(label.reshape((-1, 1)).squeeze(dim=1))
        batch_output = []
        
        for model in models[:]:
            output = model(x)
            # tuple 4 * (bs, dim)
            num, dim = output[0].shape
            output = torch.concat(output, dim=1).reshape((-1, dim))
            # output.shape (bs*4, dim)
            batch_output.append(output)
        x.detach()
        outputs.append(torch.concat(batch_output, dim=1).cpu())
        tags.append(tag[0])

outputs = torch.concat(outputs, dim=0)
# outputs.shape (4*data_size, len(models)*62)
Y = torch.concat(Y, dim=0)

# %%

preds = xgb_model.predict(outputs.detach().numpy())



preds = preds.reshape((-1, 4)).tolist()
with open('xgb_b_submission.csv', 'w') as writer:
    writer.write("num,tag\n")
    for pred, tag in zip(preds, tags):
        s = LabeltoStr(pred)
        writer.write(f"{tag},{s}\n")



#


