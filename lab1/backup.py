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
def der_sigmoid(y):
    """ First derivative of Sigmoid function.
    The input to this function should be the value that output from sigmoid function.
    """
    return y * (1 - y)

class Net:
    def __init__(self):
        self.EPS = 1e-3
        self.lr = 0.1
        self.hidden_layer_neurons = [20, 7]
        self.epochs = 40000
        self.W = [None, np.random.randn(20, 2), np.random.randn(7, 20), np.random.randn(1, 7)]
        self.b = [None, np.zeros((20, 1)), np.zeros((7, 1)), np.zeros((1, 1))]
        self.Z = [None, np.zeros((20, 1)), np.zeros((7, 1)), np.zeros((1, 1))]
        self.a = [None, np.zeros((20, 1)), np.zeros((7, 1)), np.zeros((1, 1))]
        
    
    def sigmoid(self, x):
            #這邊不能用import math的math.exp 因為他是array math指支援size-1 arrays 
        return 1/(1+np.exp(-x))

    def forward(self, inputs):
        """ Implementation of the forward pass.
        It should accepts the inputs and passing them through the network and return results.
        Args:
            inputs: (2*batch_size) ndarray
        Returns:
            output: (1*batch_size) ndarray
        """
        self.a[0]=inputs
        # hidden layer 1
        self.Z[1]=self.W[1]@self.a[0]+self.b[1]
        self.a[1]=self.sigmoid(self.Z[1])
        
        # hidden layer 2
        self.Z[2]=self.W[2]@self.a[1]+self.b[2]
        self.a[2]=self.sigmoid(self.Z[2])

        # output layer 
        self.Z[3]=self.W[3]@self.a[2]+self.b[3]
        self.a[3]=self.sigmoid(self.Z[3])

        return self.a[3]

    def calculate_loss(self, y, y_pred):
        #cross entropy: 要算第k筆data分別是0, 1的機率 -y01(log(p(0,1))) + -y11(log(p(1,1)))
        n = y.shape[1]
        loss = -(1./n) * (np.matmul(y, np.log(y_pred+self.EPS).T) + np.matmul(1-y, np.log(1-y_pred+self.EPS).T))
        return float(loss)
    

    def backward(self,gt_y,pred_y):
        """ Implementation of the backward pass.
        It should utilize the saved loss to compute gradients and update the network all the way to the front.
        Args:
            gt_y: (1*batch_size) ndarray
            pred_y: (1*batch_size) ndarray
        """
        batch_size=gt_y.shape[1]
        # bp
        grad_a3=-(gt_y/(pred_y+self.EPS)-(1-gt_y)/(1-pred_y+self.EPS))
       
        grad_z3=grad_a3*der_sigmoid(self.a[3])
        
        grad_W3=grad_z3@self.a[2].T*(1/batch_size)
        

        grad_b3=np.sum(grad_z3,axis=1,keepdims=True)*(1/batch_size)
        
        grad_a2=self.W[3].T@grad_z3
        grad_z2=grad_a2*der_sigmoid(self.a[2])
        grad_W2=grad_z2@self.a[1].T*(1/batch_size)
        grad_b2=np.sum(grad_z2,axis=1,keepdims=True)*(1/batch_size)
            
        grad_a1=self.W[2].T@grad_z2
        grad_z1=grad_a1*der_sigmoid(self.a[1])
        grad_W1=grad_z1@self.a[0].T*(1/batch_size)
        grad_b1=np.sum(grad_z1,axis=1,keepdims=True)*(1/batch_size)
        
        # update
        self.W[1]-=self.lr*grad_W1
        self.W[2]-=self.lr*grad_W2
        self.W[3]-=self.lr*grad_W3
        self.b[1]-=self.lr*grad_b1
        self.b[2]-=self.lr*grad_b2
        self.b[3]-=self.lr*grad_b3
        
        return

    # def backward(self, y):
    #     # print("backward")
    #     # print(self.Z[3].shape)
    #     # print(self.Z[2].shape)
    #     # print(self.Z[1].shape)
    #     # print(self.a[0].shape)
    #     batch_size=y.shape[1]
    #     self.grada[3] = -(np.divide(y, self.a[3]+self.eps) + np.divide(1-y, 1-self.a[3]+self.eps))
    #     self.gradZ[3] = np.multiply(self.grada[3], np.multiply(1-self.Z[3], self.Z[3])) 
    #     self.gradW[3] = np.matmul(self.gradZ[3], self.a[2].T)*(1/batch_size)
    #     self.gradb[3] = np.sum(self.gradZ[3], axis=1, keepdims=True)*(1/self.epochs)

    #     self.grada[2] = -(np.divide(y, self.a[2]+self.eps) + np.divide(1-y, 1-self.a[2]+self.eps))
    #     self.gradZ[2] = np.multiply(self.grada[2], np.multiply(1-self.Z[2], self.Z[2])) 
    #     self.gradW[2] = np.matmul(self.gradZ[2], self.a[1].T)*(1/batch_size)
    #     self.gradb[2] = np.sum(self.gradZ[2], axis=1, keepdims=True)*(1/self.epochs)

    #     self.grada[1] = -(np.divide(y, self.a[1]+self.eps) + np.divide(1-y, 1-self.a[1]+self.eps))
    #     self.gradZ[1] = np.multiply(self.grada[1], np.multiply(1-self.Z[1], self.Z[1])) 
    #     self.gradW[1] = np.matmul(self.gradZ[1], self.a[0].T)*(1/batch_size)
    #     self.gradb[1] = np.sum(self.gradZ[1], axis=1, keepdims=True)*(1/self.epochs)

        
    #     # for i in reversed(range(1, 4)):
    #     #     self.grada[i] = -(np.divide(y, self.a[i]) + np.divide(1-y, 1-self.a[i]))
    #     #     self.gradZ[i] = np.multiply(self.grada[i], np.multiply(1-self.Z[i], self.Z[i])) 
    #     #     self.gradW[i] = np.matmul(self.gradZ[i], self.a[i-1].T)
    #     #     self.gradb[i] = self.gradZ[i]

    # def update_weight(self):
        
    #     for i in range(1,4):
    #         # print(i)
    #         # print(self.gradW[i].shape)
    #         self.W[i] -= self.lr * self.gradW[i]
    #         self.b[i] -= self.lr * self.gradb[i]
           

    # def backward(self,gt_y,pred_y):
    #     """ Implementation of the backward pass.
    #     It should utilize the saved loss to compute gradients and update the network all the way to the front.
    #     Args:
    #         gt_y: (1*batch_size) ndarray
    #         pred_y: (1*batch_size) ndarray
    #     """
    #     batch_size=gt_y.shape[1]
    #     # bp
    #     grad_a3=-(gt_y/(pred_y+self.EPS)-(1-gt_y)/(1-pred_y+self.EPS))
       
    #     grad_z3=grad_a3*der_sigmoid(self.a[3])
        
    #     grad_W3=grad_z3@self.a[2].T*(1/batch_size)
        
    #     grad_b3=np.sum(grad_z3,axis=1,keepdims=True)*(1/batch_size)
        
    #     grad_a2=self.W[3].T@grad_z3
    #     grad_z2=grad_a2*der_sigmoid(self.a[2])
    #     grad_W2=grad_z2@self.a[1].T*(1/batch_size)
    #     grad_b2=np.sum(grad_z2,axis=1,keepdims=True)*(1/batch_size)
            
    #     grad_a1=self.W[2].T@grad_z2
    #     grad_z1=grad_a1*der_sigmoid(self.a[1])
    #     grad_W1=grad_z1@self.a[0].T*(1/batch_size)
    #     grad_b1=np.sum(grad_z1,axis=1,keepdims=True)*(1/batch_size)
        
    #     # update
    #     self.W[1]-=self.lr*grad_W1
    #     self.W[2]-=self.lr*grad_W2
    #     self.W[3]-=self.lr*grad_W3
    #     self.b[1]-=self.lr*grad_b1
    #     self.b[2]-=self.lr*grad_b2
    #     self.b[3]-=self.lr*grad_b3
        
    #     return

        

def generate_linear(n=100):
    data = np.random.uniform(0, 1, (n, 2))
    inputs = []
    labels = []

    for point in data:
        inputs.append([point[0], point[1]])

        if point[0] > point[1]:
            labels.append(0)
        else:
            labels.append(1)

    return np.array(inputs), np.array(labels).reshape(n, 1)



def main():
    #get data and parse them into training and testing set
    x, y = generate_linear()
    x_train, y_train, x_test, y_test = np.array(x[:50].T), np.array(y[50:].T), np.array(x[50:].T), np.array(y[50:].T)
    # print(y_train.shape)
    model = Net()
    
    #train
    for i in range(model.epochs):
        y_pred = model.forward(x_train)
        loss = model.calculate_loss(y_train, y_pred)
        model.backward(y_train, y_pred)
        # model.update_weight()

        if i % 1000 == 0:
            # print(y_pred)
            acc=(1.-np.sum(np.abs(y_train-np.round(y_pred)))/y_train.shape[1])*100
            print(f'Epochs {i}: loss={loss} accuracy={acc:.2f}%')
        

if __name__ == '__main__':
    main()