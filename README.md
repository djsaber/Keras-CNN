# Keras-CNN
基于Keras搭建一个简单的卷积神经网络CNN，用猫狗数据集和花卉数据集对CNN进行训练，完成模型的保存和加载和识别测试。<br />

环境：<br />
CUDA：11.6.134<br />
cuDNN：8.4.0<br />
keras：2.9.0<br />
tensorflow：2.9.1<br /><br />
注意：<br />
项目内目录中两个文件夹：<br />
1./datasets：保存猫狗数据集合花卉数据集<br />
2./save_model：保存训练好的模型权重文件<br /><br />
使用Keras预处理工具 ImageDataGenerator，对数据集中原始图片进行缩放，旋转等操作，以增强数据<br />
使用flow_from_directory()方法从数据集的子目录中实时生成训练和测试数据<br />
确保数据集目录格式正确：<br />
数据集<br />
---类别1<br />
-------jpg1<br />
-------png2<br />
-------bmp3<br />
-------...<br />
---类别2<br />
---类别3<br />
---...
