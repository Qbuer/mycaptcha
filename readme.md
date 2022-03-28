

|  version   | acc  |  dataset  | model |  说明  |
|  ----  | ----  | ----  | ----  | ---- |
| 8  | 73 | dataset1 | resnet18 |正常  |
| 9  | 78 | dataset1 | resnet18 |+aug  |
| 12 | 73 | dataset2 | resnet18 |正常   |
| 13 | 75 | dataset2 | resnet18 |+aug  |
| 14 | 77 | dataset1 | resnet50 |+aug |
| 32 | 70 | dataset1 | crnn     |正常 |
| 60 | 74.8 | dataset4 | resnet18 | 正常 |
| 62 | 77.4 | dataset4 | resnet18 | +aug |
| 63 | 75.9 | dataset1 |  resnet18 | 无限生成训练图片|
| 64 | 72.7 | dataset1 |  crnn | 无限生成训练图片|
| 65 | 70.3 | s/dataset1/python | resnet18 | 单独验证python验证码 |
| 67 | 79.0 | s/dataset1/other | resnet18 | 单独验证其他类型验证码 |
| 68 | 79.8 | s/dataset1/other | resnet18 | 单独验证其他类型验证码, + dropout |

# 数据说明
dataset1 2/8划分数据  
dataset2 经过图像处理  
dataset3 参了3000张python验证码  
dataset4 +生成50000张 +5000张python验证码  
数据分布
| 数据集 | python | other |
| ---   |  ---   | ---   |
| 训练集 |  8015 |  6985 |
| 测试集 |  15231 | 9769|       
|  测试B | 9798 | 5202 |
# 阶段结论
1. 实验了RES18、RES50、CRNN（CTC）、GITHUB开源解决方案，验证集ACC最高值为78%。
2. 通过测试训练集（含dataset1/aug）acc指标，acc=1.0，表明已完全拟合训练集，通过修改模型无法获得额外收益。
4. 对两种明显区别的验证码分开训练，发现python生成的验证码正确率较低。
3. 对结果进行简单融合，在单字符上多模型投票，发现结果提升了7个百分点。

# 资料

各种语言常用验证码生成库 https://blog.csdn.net/qq_41895190/article/details/102527694

# 网络结构 
resnet  https://www.jianshu.com/p/085f4c8256f1
crnn https://zhuanlan.zhihu.com/p/26078299