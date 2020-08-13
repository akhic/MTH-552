import numpy as np
import h5py
import scipy
from PIL import Image
import matplotlib.pyplot as plt
def sigmoid(x):
	return 1/(1+np.exp(x*-1))
def load():
	train = h5py.File('datasets/train_catvnoncat.h5','r')
	y_train = np.array(train['train_set_y'][:])
	x_train = np.array(train['train_set_x'][:])

	y_train = y_train.reshape((1,y_train.shape[0]))
	test = h5py.File('datasets/test_catvnoncat.h5','r')
	y_test = np.array(test['test_set_y'][:])
	x_test = np.array(test['test_set_x'][:])
	y_test = y_test.reshape((1,y_test.shape[0]))
	categories = np.array(test['list_classes'][:])

	return x_train,y_train,x_test,y_test,categories

x_train,y_train,x_test,y_test,categories = load()
#You can run these lines to check the images in our dataset
# id = 4
# plt.imshow(x_train[id])
# plt.show()
x_train = x_train.reshape((x_train.shape[0],x_train.shape[1]*x_train.shape[2]*x_train.shape[3])).T
x_test = x_test.reshape((x_test.shape[0],x_test.shape[1]*x_test.shape[2]*x_test.shape[3])).T

x_train = x_train/255.
x_test = x_test/255.

def forward_prop(X,Y,w,b):
	n = X.shape[1]
	# print(n)
	A = sigmoid(np.dot(w.T,X)+b)
	# print(A.shape)
	cost = (-1/n)*np.sum(Y*np.log(A)+(1-Y)*np.log(1-A))
	# print(cost)
	dcost_w = (1/n)*np.dot(X,(A-Y).T)
	dcost_b = (1/n)*np.sum(A-Y)

	dict_gradients = {'dcost_w':dcost_w,'dcost_b':dcost_b}
	return cost, dict_gradients

def gradient_descent(X,Y,w,b,n_iter,lr):
	# print(Y.shape)
	dict_costs = []

	for t in range(n_iter):
		cost,dict_gradients= forward_prop(X,Y,w,b)
		dcost_w = dict_gradients['dcost_w']
		dcost_b = dict_gradients['dcost_b']
		# print(dcost_w.shape)
		w = w - lr*dcost_w
		b = b - lr*dcost_b

		if t%100==0:
			dict_costs.append(cost)
			print(cost)
	return w,b,dcost_w,dcost_b,dict_costs
w = np.zeros([x_train.shape[0],1])
print(x_train.shape)
b= 0
n_iter = 2500
lr = 0.004
w,b,dcost_w,dcost_b,dict_costs = gradient_descent(x_train,y_train,w,b,n_iter,lr)


def prediction(w,b,X):
	n = X.shape[1]
	Y_p = np.zeros((1,n))
	A = sigmoid(np.dot(w.T,X)+b)
	Y_p = A>0.5
	return Y_p
y_p_test = prediction(w,b,x_test)

y_p_train = prediction(w,b,x_train)

test_p = np.mean(np.abs(y_p_test-y_test))
train_p = np.mean(np.abs(y_p_train-y_train))
print('Train set performance= '+str(100*(1-train_p))+'%')
print('Test set performance= '+str(100*(1-test_p))+'%')


def test_custom_img(img,w,b):
	image = Image.open(img)
	image.load()
	image = image.resize((64,64))
	image = np.asarray(image,dtype='float64')
	image = image/255.
	
	image = image.reshape((1, 64*64*3)).T
	predict = prediction(w,b,image)
	if(predict<1):
		return 'Its not a cat'
	else:
		return 'Its a cat'
#You can run this function to test any custom image. Just replace the name of the file with your own.
print(test_custom_img('non cat img.jpg',w,b))