from PIL import Image, ImageDraw
from pandas import read_csv, DataFrame
from random import randint
import numpy as np

def sygma(x):
    return 1/(1+np.exp(-x))

def compress_img(image_name,rectangle=None, width=None, height=None):
    img = Image.open(image_name).crop(rectangle)
    img = img.resize((width, height), Image.Resampling.LANCZOS)  
    return img

def def_x_left(test,count_layers):
    matrix_of_weights = [np.load("mod2\matrix_of_weights"+str(matr)+".npy") for matr in range(count_layers-1)]
    matrix_of_bias = [np.load("mod2\matrix_of_bias"+str(matr)+".npy") for matr in range(count_layers-1)]
    layers_sizes = [len(test)]+[len(i) for i in matrix_of_bias]
    layers=[np.zeros((layers_sizes[i],1)) for i in range(count_layers)]

    layers[0]=test.reshape((-1,1))
    for k in range(1,count_layers):
        layers[k]=sygma(matrix_of_weights[k-1] @ layers[k-1]+matrix_of_bias[k-1])
    return layers[-1][0][0]*1000

def def_y_left(test,count_layers):
    matrix_of_weights = [np.load("mod4\matrix_of_weights"+str(matr)+".npy") for matr in range(count_layers-1)]
    matrix_of_bias = [np.load("mod4\matrix_of_bias"+str(matr)+".npy") for matr in range(count_layers-1)]

    layers_sizes = [len(test)]+[len(i) for i in matrix_of_bias]
    layers=[np.zeros((layers_sizes[i],1)) for i in range(count_layers)]

    layers[0]=test.reshape((-1,1))
    for k in range(1,count_layers):
        layers[k]=sygma(matrix_of_weights[k-1] @ layers[k-1]+matrix_of_bias[k-1])
    return layers[-1][0][0]*1000

def def_x_right(test,count_layers):
    matrix_of_weights = [np.load("mod3\matrix_of_weights"+str(matr)+".npy") for matr in range(count_layers-1)]
    matrix_of_bias = [np.load("mod3\matrix_of_bias"+str(matr)+".npy") for matr in range(count_layers-1)]

    layers_sizes = [len(test)]+[len(i) for i in matrix_of_bias]
    layers=[np.zeros((layers_sizes[i],1)) for i in range(count_layers)]

    layers[0]=test.reshape((-1,1))
    for k in range(1,count_layers):
        layers[k]=sygma(matrix_of_weights[k-1] @ layers[k-1]+matrix_of_bias[k-1])
    return layers[-1][0][0]*1000

def def_y_right(test,count_layers):
    matrix_of_weights = [np.load("mod5\matrix_of_weights"+str(matr)+".npy") for matr in range(count_layers-1)]
    matrix_of_bias = [np.load("mod5\matrix_of_bias"+str(matr)+".npy") for matr in range(count_layers-1)]

    layers_sizes = [len(test)]+[len(i) for i in matrix_of_bias]
    layers=[np.zeros((layers_sizes[i],1)) for i in range(count_layers)]

    layers[0]=test.reshape((-1,1))
    for k in range(1,count_layers):
        layers[k]=sygma(matrix_of_weights[k-1] @ layers[k-1]+matrix_of_bias[k-1])
    return layers[-1][0][0]*1000

def define_lokation(path,width,height):
    test = np.asarray(compress_img(path,None,50,50)).reshape(50*50*3)/1000
    return [round(def_x_left(test,3)/50*width),round(def_y_left(test,3)/50*height),round(def_x_right(test,3)/50*width),round(def_y_right(test,3)/50*height)]

def def_sign(path,width,height):
    count_layers = 3
    location = define_lokation(Datatest.Path[number_test],width,height)

    test1=np.asarray(compress_img(path,location,60,60)).reshape(3600,3)/255
    test=np.array([sum(j)/3 for j in test1])

    matrix_of_weights = [np.load("mod1\matrix_of_weights"+str(matr)+".npy") for matr in range(count_layers-1)]
    matrix_of_bias = [np.load("mod1\matrix_of_bias"+str(matr)+".npy") for matr in range(count_layers-1)]

    layers_sizes = [len(test)]+[len(i) for i in matrix_of_bias]
    layers=[np.zeros((layers_sizes[i],1)) for i in range(count_layers)]

    layers[0]=test.reshape((-1,1))
    for k in range(1,count_layers):
        layers[k]=sygma(matrix_of_weights[k-1] @ layers[k-1]+matrix_of_bias[k-1])

    return (np.argmax(layers[-1]))

Datatest = read_csv("Test.csv")
number_test=randint(0,len(Datatest))

class_sign = def_sign(Datatest.Path[number_test],Datatest.Width[number_test],Datatest.Height[number_test])
location = define_lokation(Datatest.Path[number_test],Datatest.Width[number_test],Datatest.Height[number_test])

shape = [(location[0],location[1]),(location[2],location[3])]

img = Image.open(Datatest.Path[number_test])
img1 = ImageDraw.Draw(img)
img1.rectangle(shape,outline="red",width=1)
img.show()

img_meta=Image.open("Meta\\"+str(class_sign)+".png")
img_meta.show()