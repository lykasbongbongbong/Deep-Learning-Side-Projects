'''
class Net:
    #initialization
    W: [None, shape(2, 20), shape(20, 7), shape(1, 7)]
    b: [None, (20,1), (7,1), (1,1)]
    Z: [None, (20,1), (7,1), (1,1)]
    a: [None, (20,1), (7,1), (1,1)]
    learning_rate
    hidden_layer_neuron
    iterations
    main: 
        get data:
            linear / XOR
            divide them into training and testing set (50, 50)
            reshape
    #forward
    #backward
train: 
    forward
    loss 
    backward
    update_weight

test:
    forward
    loss
'''
import numpy as np

class Net:
    def __init__(self):
        self.lr = 0.008
        self.hidden_layer_neurons = [20, 7, 1]
        self.epochs = 10000
        self.W = [None, np.random.randn(2, 20), np.random.randn(20, 7), np.random.randn(7, 1)]
        self.b = [None, np.random.randn(20, 1), np.random.randn(7, 1), np.random.randn(1, 1)]
        self.Z = [None, np.random.randn(20, 1), np.random.randn(7, 1), np.random.randn(1, 1)]
        self.a = [None, np.random.randn(20, 1), np.random.randn(7, 1), np.random.randn(1, 1)]
    
    
    def sigmoid(self, x):
            #這邊不能用import math的math.exp 因為他是array math指支援size-1 arrays 
        return 1/(1+np.exp(-x))

    def forward(self, x, y):
        self.a[0] = x
        #1st hidden layer
        self.Z[1] = np.matmul(self.W[1].T, self.a[0]) + self.b[1]
        self.a[1] = self.sigmoid(self.Z[1])

        #2nd hidden layer
        self.Z[2] = np.matmul(self.W[2].T, self.a[1]) + self.b[2]
        self.a[2] = self.sigmoid(self.Z[2])

        #3rd output layer
        self.Z[3] = np.matmul(self.W[3].T, self.a[2]) + self.b[2]
        self.a[3] = self.sigmoid(self.Z[3])

    
    def calculate_loss(self):
        #cross entropy
        print("aaaaa")
        print(self.a[3])
        

def generate_linear(n=100):
    pts = np.random.uniform(0,1,(n,2))  
    inputs = []
    labels = []
    for pt in pts:
        inputs.append([pt[0], pt[1]])
        if pt[0] > pt[1]:
            labels.append(0)
        else:
            labels.append(1)
    return np.array(inputs), np.array(labels).reshape(n,1)



def main():
    #get data and parse them into training and testing set
    x, y = generate_linear()
    x_train, y_train, x_test, y_test = np.array(x[:50].T), np.array(y[50:].T), np.array(x[50:].T), np.array(y[50:].T)


    model = Net()
    
    #train
    for i in range(model.epochs):
        model.forward(x_train, y_train)
        model.calculate_loss()
        # model.backward()
        # model.update_weight()




if __name__ == '__main__':
    main()