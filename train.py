#coding=gbk

from model import Simple_CNN
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator


data_choose = input("使用哪个数据集来训练CNN？（cat_vs_dog：1 / flower_photos：2）\n")
if data_choose != "1" and data_choose != "2": raise ValueError("输入错误")

#------------------------------设置参数----------------------------------------
input_shape = (128, 128, 3 )
if data_choose == "1" : output_dim = 2
if data_choose == "2" : output_dim = 5
batch_size = 32           
epochs = 20
#-----------------------------------------------------------------------------


#------------------------数据集路径、模型保存路径---------------------------------
if data_choose == "1":
    data_path = "D:/科研/python代码/炼丹手册/CNN/datasets/cat_vs_dog"
    save_path = "D:/科研/python代码/炼丹手册/CNN/save_models/cnn_catvsdog.h5"
    
if data_choose == "2":
    data_path = "D:/科研/python代码/炼丹手册/CNN/datasets/flower_photos"
    save_path = "D:/科研/python代码/炼丹手册/CNN/save_models/cnn_flowers.h5"

train_path = data_path + "/train"
valid_path = data_path + "/valid"
test_path = data_path + "/test"
#-----------------------------------------------------------------------------


#-----------------------使用Keras数据生成器工具自动处理图片------------------------
train_datagen = ImageDataGenerator(
        rotation_range=10,         # 旋转范围
        width_shift_range=0.1,     # 宽度偏移
        height_shift_range=0.1,    # 高度偏移
        shear_range=0.15,          # 绝对偏移
        channel_shift_range=10,    # 通道移动范围
        rescale=1./255,            # 重缩放因子
        zoom_range=0.2,            # 随机缩放范围
        horizontal_flip=True       # 随机水平翻转
        )

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
        train_path,
        target_size=(input_shape[0], input_shape[1]),
        batch_size=batch_size,
        class_mode='categorical')

validation_generator = test_datagen.flow_from_directory(
        valid_path,
        target_size=(input_shape[0], input_shape[1]),
        batch_size=batch_size,
        class_mode='categorical')
#-----------------------------------------------------------------------------


#-----------------------------构建简单的cnn模型----------------------------------
cnn = Simple_CNN(
    input_shape=input_shape,
    output_dim=output_dim
    )
cnn.build((None,128,128,3))
cnn.summary()
cnn.compile(
    optimizer=Adam(learning_rate=0.001), 
    loss='categorical_crossentropy', 
    metrics=['acc'])
#-----------------------------------------------------------------------------


#--------------------------------训练和保存-------------------------------------
cnn.fit(
    x=train_generator,
    epochs=epochs,
    steps_per_epoch=30,
    validation_data=validation_generator,
    validation_steps=5
    )
cnn.save_weights(save_path)
#-----------------------------------------------------------------------------