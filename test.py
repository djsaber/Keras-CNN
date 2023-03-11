#coding=gbk

from model import Simple_CNN
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator


data_choose = input("使用哪个数据集来测试CNN？（cat_vs_dog：1 / flower_photos：2）\n")
if data_choose != "1" and data_choose != "2": raise ValueError("输入错误")


#------------------------------设置参数----------------------------------------
input_shape = (128, 128, 3)
if data_choose == "1":
    output_dim = 2
    batch_size = 50   
if data_choose == "2":
    output_dim = 5
    batch_size = 5           
#-----------------------------------------------------------------------------


#------------------------数据集路径、模型保存路径---------------------------------
if data_choose == "1":
    data_path = "D:/科研/python代码/炼丹手册/CNN/datasets/cat_vs_dog"
    save_path = "D:/科研/python代码/炼丹手册/CNN/save_models/cnn_catvsdog.h5"
    
if data_choose == "2":
    data_path = "D:/科研/python代码/炼丹手册/CNN/datasets/flower_photos"
    save_path = "D:/科研/python代码/炼丹手册/CNN/save_models/cnn_flowers.h5"

test_path = data_path + "/test"
#-----------------------------------------------------------------------------


#-----------------------使用Keras数据生成器工具自动处理图片------------------------
test_datagen = ImageDataGenerator(rescale=1./255)

test_generator = test_datagen.flow_from_directory(
    test_path,
    target_size=(input_shape[0], input_shape[1]),
    batch_size=batch_size,
    class_mode='categorical')
#-----------------------------------------------------------------------------


#-----------------------------构建模型、加载权重---------------------------------
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
cnn.load_weights(save_path)
#-----------------------------------------------------------------------------


#--------------------------------训练和保存-------------------------------------
cnn.evaluate(
    x=test_generator,
    steps=1,
    )
#-----------------------------------------------------------------------------