# Copyright 2021 Zhongyang Zhang
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
from pathlib2 import Path

def load_model_path(v_num, best=True):
    """ When best = True, return the best model's path in a directory 
        by selecting the best model with largest epoch. If not, return
        the last model saved. You must provide at least one of the 
        first three args.
    Args: 
        root: The root directory of checkpoints. It can also be a
            model ckpt file. Then the function will return it.
        version: The name of the version you are going to load.
        v_num: The version's number that you are going to load.
        best: Whether return the best model.
    """
    def sort_by_epoch(path):
        name = path.stem
        epoch=int(name.split('-')[1].split('=')[1])
        return epoch
    
    assert v_num is not None

    root = str(Path('lightning_logs', f'version_{v_num}', 'checkpoints'))

    if best:
        files=[i for i in list(Path(root).iterdir()) if i.stem.startswith('best')]
        # files.sort(key=sort_by_epoch, reverse=True)
        res = str(files[0])
    else:
        res = str(Path(root) / 'last.ckpt')
    return res

def load_model_path_by_args(args):
    return load_model_path(args.vnum)