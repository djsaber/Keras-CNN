#coding=gbk

from model import Simple_CNN
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator


data_choose = input("ʹ���ĸ����ݼ���ѵ��CNN����cat_vs_dog��1 / flower_photos��2��\n")
if data_choose != "1" and data_choose != "2": raise ValueError("�������")

#------------------------------���ò���----------------------------------------
input_shape = (128, 128, 3 )
if data_choose == "1" : output_dim = 2
if data_choose == "2" : output_dim = 5
batch_size = 32           
epochs = 20
#-----------------------------------------------------------------------------


#------------------------���ݼ�·����ģ�ͱ���·��---------------------------------
if data_choose == "1":
    data_path = "D:/����/python����/�����ֲ�/CNN/datasets/cat_vs_dog"
    save_path = "D:/����/python����/�����ֲ�/CNN/save_models/cnn_catvsdog.h5"
    
if data_choose == "2":
    data_path = "D:/����/python����/�����ֲ�/CNN/datasets/flower_photos"
    save_path = "D:/����/python����/�����ֲ�/CNN/save_models/cnn_flowers.h5"

train_path = data_path + "/train"
valid_path = data_path + "/valid"
test_path = data_path + "/test"
#-----------------------------------------------------------------------------


#-----------------------ʹ��Keras���������������Զ�����ͼƬ------------------------
train_datagen = ImageDataGenerator(
        rotation_range=10,         # ��ת��Χ
        width_shift_range=0.1,     # ���ƫ��
        height_shift_range=0.1,    # �߶�ƫ��
        shear_range=0.15,          # ����ƫ��
        channel_shift_range=10,    # ͨ���ƶ���Χ
        rescale=1./255,            # ����������
        zoom_range=0.2,            # ������ŷ�Χ
        horizontal_flip=True       # ���ˮƽ��ת
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


#-----------------------------�����򵥵�cnnģ��----------------------------------
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


#--------------------------------ѵ���ͱ���-------------------------------------
cnn.fit(
    x=train_generator,
    epochs=epochs,
    steps_per_epoch=30,
    validation_data=validation_generator,
    validation_steps=5
    )
cnn.save_weights(save_path)
#-----------------------------------------------------------------------------