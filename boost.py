# %%
from model.res18 import Res18Model
from model.res50 import Res50Model
from model.crnn import CrnnModel
from dataModule import captchaDataModule
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
data = captchaDataModule('data/dataset4', batch_size=16, num_workers=4)
data.setup()

# %%
dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
models = [model.to(dev) for model in models]
print(dev)

# %%
outputs = []
Y = []
with torch.no_grad():
    for x, label, _ in tqdm(data.train_dataloader()):
        
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
        outputs.append(torch.concat(batch_output, dim=1).cpu())
        x.detach()

outputs = torch.concat(outputs, dim=0)
# outputs.shape (4*data_size, len(models)*62)
Y = torch.concat(Y, dim=0)

# %%

xgb_X = outputs.detach().numpy()
xgb_Y = Y.detach().numpy()

# %%
xgb_model = xgb.XGBClassifier()
xgb_model.fit(xgb_X, xgb_Y)

# %% [markdown]
# 

# %%
outputs = []
Y = []
with torch.no_grad():
    for x, label, _ in tqdm(data.val_dataloader()):
        
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

outputs = torch.concat(outputs, dim=0)
# outputs.shape (4*data_size, len(models)*62)
Y = torch.concat(Y, dim=0)

# %%

pred = xgb_model.predict(outputs.detach().numpy())

# %%
pred = torch.tensor(pred)
pred = pred.reshape((-1, 4))
xgb_val_Y = Y.reshape((-1, 4))


# %%
diff = (xgb_val_Y != pred)
diff = diff.sum(1)
diff = diff != 0
wrong_num = diff.sum(0).item()
num = xgb_val_Y.shape[0]
acc = (num - wrong_num) / num

# %%
print(f"acc: {acc}")

# %%
xgb_model.save_model('xgb.model')


