import numpy as np
from numpy import genfromtxt
import random as rd

# Managing the data 
iris_data = genfromtxt('iris.csv', delimiter=',')
X=np.delete(iris_data,-1,1)

#Encode The Output Variable Y
a,b,c =np.array([1,0,0]),np.array((0,1,0)),np.array((0,0,1))
a,b,c =np.tile(a,(50,1)),np.tile(b,(50,1)), np.tile(c,(50,1))
#Y=np.concatenate((a.T,b.T,c.T),axis=1)
Y=np.concatenate((a,b,c))


#Randomly initialize weights
#np.random.seed(1000)
W1 = 2*np.random.randn(4, 5)-1
W2 = 2*np.random.randn(5, 3)-1
theta = W1,W2

# Sigmoid function
def sigmoid(x,derivate=False):
	if derivate == True:
		return 1/(1+np.exp(-x))*(1-(1/(1+np.exp(-x))))
	else:
		return 1 / (1 + np.e ** -x)
		
# Basic parameters
epoch = 60000
learning_rate = 0.1
lamb = .1 # lambda for regularization

#Cost function
def cost_function(X,Y,h_theta, m=len(X)):
	return - (np.sum((Y*np.log(h_theta))+(1-Y)*np.log(1-h_theta))/m)
	# Regulariation
	#+ (lamb/m * (np.sum(W1)**2 + np.sum(W2)**2))

#Cost function derivate
def cost_function_derivate(X,Y,h_theta):
	return np.sum(h_theta-Y)*X


#Backward Propagation
for i in range(epoch):
	print("epoch left :",epoch-i)
	
	#Forward Propagation / da3 and da2 are respectively the derivative of the sigmoid a3 and a2
	a2=sigmoid(np.dot(X,W1))
	a3=sigmoid(np.dot(a2,W2))
	da2=sigmoid(np.dot(X,W1),True)
	da3=sigmoid(np.dot(a2,W2),True)

	#defining error - delta
	delta3 = (Y-a3) * sigmoid(a3,True)
	delta2 = np.dot(delta3,W2.T)* sigmoid(a2,True)

	#Sommation of the deltas error
	sum_delta3, sum_delta2 = 0,0
	sum_delta3 += np.dot(a2.T,delta3)
	sum_delta2 += np.dot(X.T,delta2)

	#Update the weights
	W2 += learning_rate * (sum_delta3/len(X))
	W1 += learning_rate * (sum_delta2/len(X))

print("cf",cost_function(X,Y,a3,len(X)))

#Gradient Check -TO FIX-
def Gradient_Chek():
	epsilon = 0.0001
	W1_e_p = W1+epsilon
	a2_p=sigmoid(np.dot(X,W1_e_p))
	a3_p=sigmoid(np.dot(a2_p,W2))

	W1_e_m = W1-epsilon
	a2_m=sigmoid(np.dot(X,W1_e_m))
	a3_m=sigmoid(np.dot(a2_m,W2))
	J_W1 = (cost_function(X,Y,a3_p,len(X))-cost_function(X,Y,a3_m,len(X)))/(2*epsilon)

	W2_e_p = W2+epsilon
	a2_p=sigmoid(np.dot(X,W1))
	a3_p=sigmoid(np.dot(a2,W2_e_p))

	W2_e_m = W2-epsilon
	a2_m=sigmoid(np.dot(X,W1))
	a3_m=sigmoid(np.dot(a2,W2_e_m))
	J_W2 = (cost_function(X,Y,a3_p,len(X))-cost_function(X,Y,a3_m,len(X)))/(2*epsilon)

	d2=cost_function_derivate(a2,Y,a3)
	d1= cost_function_derivate(X,Y,a3)

	GD=(J_W2+J_W1)-(np.dot(d1.T,d2))
	print(GD<=epsilon)
	print(GD)



print("accuracy : ",100-np.sum(a3-Y))


