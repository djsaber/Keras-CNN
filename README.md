# Keras-CNN
基于Keras搭建一个简单的卷积神经网络CNN，用猫狗数据集和花卉数据集对CNN进行训练，完成模型的保存和加载和识别测试。<br />

环境：<br />
CUDA：11.6.134<br />
cuDNN：8.4.0<br />
keras：2.9.0<br />
tensorflow：2.9.1<br /><br />
注意：<br />
项目内目录中两个文件夹：<br />
1. /datasets：保存数据集文件<br />
2. /save_model：保存训练好的模型权重文件<br />
3. 使用Keras预处理工具 ImageDataGenerator，对数据集中原始图片进行缩放，旋转等操作，以增强数据<br />
4. 使用flow_from_directory()方法从数据集的子目录中实时生成训练和测试数据<br /><br />

准备了两种数据集：<br />
1. cat_vs_dog：猫狗数据集,训练集/验证集/测试集包含500/100/50张猫和狗图片 <br />
链接：https://pan.baidu.com/s/1bcXBZy43KgUMlsqdjfjB5g?pwd=52dl 提取码：52dl<br />
3. flower_photos：花卉数据集，训练集/验证集/测试集包含约~600/60/5张五种花卉图片 <br />
链接：https://pan.baidu.com/s/1yobCV9j9m2la12YTId1TJg?pwd=52dl 提取码：52dl<br /><br />

使用其他数据集时确保数据集目录格式正确：<br />
数据集文件夹<br />
---类别1文件夹<br />
&emsp;---1.jpg<br />
&emsp;---2.png<br />
&emsp;---3.bmp<br />
&emsp;--- ...<br />
---类别2文件夹<br />
---类别3文件夹<br />
--- ...
