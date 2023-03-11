#coding=gbk

from model import Simple_CNN
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator


data_choose = input("ʹ���ĸ����ݼ�������CNN����cat_vs_dog��1 / flower_photos��2��\n")
if data_choose != "1" and data_choose != "2": raise ValueError("�������")


#------------------------------���ò���----------------------------------------
input_shape = (128, 128, 3)
if data_choose == "1":
    output_dim = 2
    batch_size = 50   
if data_choose == "2":
    output_dim = 5
    batch_size = 5           
#-----------------------------------------------------------------------------


#------------------------���ݼ�·����ģ�ͱ���·��---------------------------------
if data_choose == "1":
    data_path = "D:/����/python����/�����ֲ�/CNN/datasets/cat_vs_dog"
    save_path = "D:/����/python����/�����ֲ�/CNN/save_models/cnn_catvsdog.h5"
    
if data_choose == "2":
    data_path = "D:/����/python����/�����ֲ�/CNN/datasets/flower_photos"
    save_path = "D:/����/python����/�����ֲ�/CNN/save_models/cnn_flowers.h5"

test_path = data_path + "/test"
#-----------------------------------------------------------------------------


#-----------------------ʹ��Keras���������������Զ�����ͼƬ------------------------
test_datagen = ImageDataGenerator(rescale=1./255)

test_generator = test_datagen.flow_from_directory(
    test_path,
    target_size=(input_shape[0], input_shape[1]),
    batch_size=batch_size,
    class_mode='categorical')
#-----------------------------------------------------------------------------


#-----------------------------����ģ�͡�����Ȩ��---------------------------------
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


#--------------------------------ѵ���ͱ���-------------------------------------
cnn.evaluate(
    x=test_generator,
    steps=1,
    )
#-----------------------------------------------------------------------------