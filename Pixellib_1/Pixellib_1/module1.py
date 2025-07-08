import numpy as np
from pandas import read_csv
from sklearn.utils import shuffle
from PIL import Image

def sygma(x):
    return 1/(1+np.exp(-x))

def prim_sygma(x):
    return x*(1-x)

def compress_img(image_name,rectangle, width=None, height=None):
    img = (Image.open(image_name)).crop(rectangle)
    img = img.resize((width, height), Image.Resampling.LANCZOS)  
    return img

def learn_define_sign(train, labels, dim):
	layers_sizes=[dim,150,len(labels[0])]
	count_layers=len(layers_sizes)
	layers=[]
	epochs=5
	step_learning=0.01

	matrix_of_weights=[np.random.uniform(1,-1,(layers_sizes[i],layers_sizes[i-1])) for i in range(1,count_layers)]
	matrix_of_bias=[np.zeros((layers_sizes[i],1)) for i in range(1,count_layers)]

	layers=[np.zeros((layers_sizes[i],1)) for i in range(count_layers)]

	for epoch in range(epochs):
		e_correct=0
	
		for vector,label in zip(train,labels):

			layers[0]=vector.reshape((-1,1))
			label=label.reshape((-1,1))
		
			for k in range(1,count_layers):
				layers[k]=sygma(matrix_of_weights[k-1] @ layers[k-1]+matrix_of_bias[k-1])
			
			err=2*(layers[-1]-label)
		
			e_correct+=int(np.argmax(layers[-1])==np.argmax(label))
		
			for k in range(count_layers-2,-1,-1):
				err1=err
				err=np.transpose(matrix_of_weights[k])@ err * prim_sygma(layers[k])
				matrix_of_weights[k]-=step_learning*err1 @ np.transpose(layers[k])
				matrix_of_bias[k]-=step_learning*err1
		   		
		   	
		print(epoch,":",round((e_correct/len(train))*100,3))

	for matr in range(len(matrix_of_weights)):
		np.save("mod1\matrix_of_weights"+str(matr),matrix_of_weights[matr])
	for matr in range(len(matrix_of_bias)):
		np.save("mod1\matrix_of_bias"+str(matr),matrix_of_bias[matr])
	
def learn_define_X_left(train, labels, dim):
	print("learn_define_X_left")
	layers_sizes=[dim,100,1]
	count_layers=len(layers_sizes)
	layers=[]
	epochs=3
	epsilon=3
	step_learning=0.008

	matrix_of_weights=[np.random.uniform(-0.05,0.05,(layers_sizes[i],layers_sizes[i-1])) for i in range(1,count_layers)]
	matrix_of_bias=[np.zeros((layers_sizes[i],1)) for i in range(1,count_layers)]
	layers=[np.zeros((layers_sizes[i],1)) for i in range(count_layers)]

	for epoch in range(epochs):
		e_correct=0
	
		for vector,label in zip(train,labels):

			layers[0]=vector.reshape((-1,1))
		
			for k in range(1,count_layers):
				layers[k]=sygma(matrix_of_weights[k-1] @ layers[k-1]+matrix_of_bias[k-1])
			
			err=2*(layers[-1]-label)
		
			e_correct+=int(abs(layers[-1][0][0] - label)*1000<epsilon)
		
			for k in range(count_layers-2,-1,-1):
		   		matrix_of_weights[k]-=step_learning*err @ np.transpose(layers[k])
		   		matrix_of_bias[k]-=step_learning*err
		   		err=np.transpose(matrix_of_weights[k])@ err * prim_sygma(layers[k])
		   	
		print(epoch,":",round((e_correct/len(train))*100,3))

	for matr in range(len(matrix_of_weights)):
		np.save("mod2\matrix_of_weights"+str(matr),matrix_of_weights[matr])
	for matr in range(len(matrix_of_bias)):
		np.save("mod2\matrix_of_bias"+str(matr),matrix_of_bias[matr])

def learn_define_X_right(train, labels, dim):
	print("learn_define_X_right")
	layers_sizes=[dim,100,1]
	count_layers=len(layers_sizes)
	layers=[]
	epochs=3
	epsilon=3
	step_learning=0.008

	matrix_of_weights=[np.random.uniform(-0.05,0.05,(layers_sizes[i],layers_sizes[i-1])) for i in range(1,count_layers)]
	matrix_of_bias=[np.zeros((layers_sizes[i],1)) for i in range(1,count_layers)]
	layers=[np.zeros((layers_sizes[i],1)) for i in range(count_layers)]

	for epoch in range(epochs):
		e_correct=0
	
		for vector,label in zip(train,labels):

			layers[0]=vector.reshape((-1,1))
		
			for k in range(1,count_layers):
				layers[k]=sygma(matrix_of_weights[k-1] @ layers[k-1]+matrix_of_bias[k-1])
			
			err=2*(layers[-1]-label)
		
			e_correct+=int(abs(layers[-1][0][0] - label)*1000<epsilon)
		
			for k in range(count_layers-2,-1,-1):
		   		matrix_of_weights[k]-=step_learning*err @ np.transpose(layers[k])
		   		matrix_of_bias[k]-=step_learning*err
		   		err=np.transpose(matrix_of_weights[k])@ err * prim_sygma(layers[k])
		   	
		print(epoch,":",round((e_correct/len(train))*100,3))

	for matr in range(len(matrix_of_weights)):
		np.save("mod3\matrix_of_weights"+str(matr),matrix_of_weights[matr])
	for matr in range(len(matrix_of_bias)):
		np.save("mod3\matrix_of_bias"+str(matr),matrix_of_bias[matr])

def learn_define_Y_left(train, labels, dim):
	print("learn_define_Y_left")
	layers_sizes=[dim,100,1]
	count_layers=len(layers_sizes)
	layers=[]
	epochs=3
	epsilon=3
	step_learning=0.008

	matrix_of_weights=[np.random.uniform(-0.05,0.05,(layers_sizes[i],layers_sizes[i-1])) for i in range(1,count_layers)]
	matrix_of_bias=[np.zeros((layers_sizes[i],1)) for i in range(1,count_layers)]
	layers=[np.zeros((layers_sizes[i],1)) for i in range(count_layers)]

	for epoch in range(epochs):
		e_correct=0
	
		for vector,label in zip(train,labels):

			layers[0]=vector.reshape((-1,1))
		
			for k in range(1,count_layers):
				layers[k]=sygma(matrix_of_weights[k-1] @ layers[k-1]+matrix_of_bias[k-1])
			
			err=2*(layers[-1]-label)
		
			e_correct+=int(abs(layers[-1][0][0] - label)*1000<epsilon)
		
			for k in range(count_layers-2,-1,-1):
		   		matrix_of_weights[k]-=step_learning*err @ np.transpose(layers[k])
		   		matrix_of_bias[k]-=step_learning*err
		   		err=np.transpose(matrix_of_weights[k])@ err * prim_sygma(layers[k])
		   	
		print(epoch,":",round((e_correct/len(train))*100,3))

	for matr in range(len(matrix_of_weights)):
		np.save("mod4\matrix_of_weights"+str(matr),matrix_of_weights[matr])
	for matr in range(len(matrix_of_bias)):
		np.save("mod4\matrix_of_bias"+str(matr),matrix_of_bias[matr])

def learn_define_Y_right(train, labels, dim):
	print("learn_define_Y_right")
	layers_sizes=[dim,100,1]
	count_layers=len(layers_sizes)
	layers=[]
	epochs=3
	epsilon=3
	step_learning=0.008

	matrix_of_weights=[np.random.uniform(-0.05,0.05,(layers_sizes[i],layers_sizes[i-1])) for i in range(1,count_layers)]
	matrix_of_bias=[np.zeros((layers_sizes[i],1)) for i in range(1,count_layers)]
	layers=[np.zeros((layers_sizes[i],1)) for i in range(count_layers)]

	for epoch in range(epochs):
		e_correct=0
	
		for vector,label in zip(train,labels):

			layers[0]=vector.reshape((-1,1))
		
			for k in range(1,count_layers):
				layers[k]=sygma(matrix_of_weights[k-1] @ layers[k-1]+matrix_of_bias[k-1])
			
			err=2*(layers[-1]-label)
		
			e_correct+=int(abs(layers[-1][0][0] - label)*1000<epsilon)
		
			for k in range(count_layers-2,-1,-1):
		   		matrix_of_weights[k]-=step_learning*err @ np.transpose(layers[k])
		   		matrix_of_bias[k]-=step_learning*err
		   		err=np.transpose(matrix_of_weights[k])@ err * prim_sygma(layers[k])
		   	
		print(epoch,":",round((e_correct/len(train))*100,3))

	for matr in range(len(matrix_of_weights)):
		np.save("mod5\matrix_of_weights"+str(matr),matrix_of_weights[matr])
	for matr in range(len(matrix_of_bias)):
		np.save("mod5\matrix_of_bias"+str(matr),matrix_of_bias[matr])

data = dict(read_csv("Train.csv"))
width,height=60,60
rectangles = zip(data["Roi.X1"],data["Roi.Y1"],data["Roi.X2"],data["Roi.Y2"])
train = [np.asarray(compress_img(i,rectangle,width,height)).reshape(width*height,3)/255 for i,rectangle in zip(data["Path"],rectangles)]
train1=[np.array([sum(j)/3 for j in i]) for i in train]
labels=data["ClassId"]
print(1)

labels,train1=shuffle(labels,train1)
labels=np.eye(43)[labels]
learn_define_sign(train1,labels,width*height)

"""
data = dict(read_csv("Train.csv"))
width,height=50,50
train = [np.asarray(compress_img(i,None,width,height)).reshape(width*height*3)/1000 for i in data["Path"]]

print(1)

Roi_X1 = np.array([X_left/size for X_left,size in zip(data["Roi.X1"],data["Width"])])*width/1000
Roi_Y1 = np.array([X_left/size for X_left,size in zip(data["Roi.Y1"],data["Height"])])*height/1000
Roi_X2 = np.array([X_left/size for X_left,size in zip(data["Roi.X2"],data["Width"])])*width/1000
Roi_Y2 = np.array([X_left/size for X_left,size in zip(data["Roi.Y2"],data["Height"])])*height/1000

learn_define_X_left(train, Roi_X1, width*height*3)
learn_define_X_right(train, Roi_X2, width*height*3)
learn_define_Y_left(train, Roi_Y1, width*height*3)
learn_define_Y_right(train, Roi_Y2, width*height*3)
"""
