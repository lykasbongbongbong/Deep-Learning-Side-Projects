import numpy as np
import matplotlib.pyplot as plt


class Net:
    def __init__(self):
        self.linear = False
        self.XOR = False

        self.lr = 0.8
        self.hln = [3, 3, 1]
        self.epochs = 10000
        self.eps = 1e-8
        self.steplr_step = 500
        self.output_loss_interval = 100
        
       
        #momentum param
        self.vt = [None, np.zeros((self.hln[0], 2)), np.zeros((self.hln[1], self.hln[0])), np.zeros((1, self.hln[1]))]
        self.vt2 = [None, np.zeros((self.hln[0], 1)), np.zeros((self.hln[1], 1)), np.zeros((1, 1))]
        
        
        self.W = [None, np.random.randn(self.hln[0], 2), np.random.randn(self.hln[1], self.hln[0]), np.random.randn(1, self.hln[1])]
        self.b = [None, np.random.randn(self.hln[0], 1), np.random.randn(self.hln[1], 1), np.random.randn(1, 1)]
        self.Z = [None, np.random.randn(self.hln[0], 1), np.random.randn(self.hln[1], 1), np.random.randn(1, 1)]
        self.a = [None, np.random.randn(self.hln[0], 1), np.random.randn(self.hln[1], 1), np.random.randn(1, 1)]
        self.gradW = [None, np.random.randn(self.hln[0], 2), np.random.randn(self.hln[1], self.hln[0]), np.random.randn(1, self.hln[1])]
        self.gradb = [None, np.random.randn(self.hln[0], 1), np.random.randn(self.hln[1], 1), np.random.randn(1, 1)]
        self.gradZ = [None, np.random.randn(self.hln[0], 1), np.random.randn(self.hln[1], 1), np.random.randn(1, 1)]
        self.grada = [None, np.random.randn(self.hln[0], 1), np.random.randn(self.hln[1], 1), np.random.randn(1, 1)]
    
    def sigmoid(self, x):
            #這邊不能用import math的math.exp 因為他是array math指支援size-1 arrays 
        return 1.0/(1.0+np.exp(-x))
    
    def derivative_sigmoid(self, x):
        return np.multiply(x, 1.0-x)

    def forward(self, x, y):
        self.a[0] = x
        #1st hidden layer
        self.Z[1] = np.matmul(self.W[1], self.a[0]) + self.b[1]
        self.a[1] = self.sigmoid(self.Z[1])

        #2nd hidden layer
        self.Z[2] = np.matmul(self.W[2], self.a[1]) + self.b[2]
        self.a[2] = self.sigmoid(self.Z[2])

        #3rd output layer
        self.Z[3] = np.matmul(self.W[3], self.a[2]) + self.b[3]
        self.a[3] = self.sigmoid(self.Z[3]) 
        return self.a[3]

    def calculate_loss(self, y, y_pred):
        #cross entropy: 要算第k筆data分別是0, 1的機率 -y01(log(p(0,1))) + -y11(log(p(1,1)))
        n = y.shape[1]
        # print((-np.matmul(y, np.log(y_pred).T)))
        
        loss = 1./n * (-np.matmul(y, np.log(y_pred).T) - np.matmul(1-y, np.log(1-y_pred).T))
        return float(loss)
    
    def backward(self, y, pred_y):
        #no work(need refinement)
        # 
        
        batch_size=y.shape[1]
        
        self.grada[3] = -(np.divide(y, pred_y+self.eps) - np.divide(1-y, 1-pred_y+self.eps))
        self.gradZ[3] = np.multiply(self.grada[3], self.derivative_sigmoid(self.a[3])) 
        self.gradW[3] = np.matmul(self.gradZ[3], self.a[2].T)*(1/batch_size)
        self.gradb[3] = np.sum(self.gradZ[3], axis=1, keepdims=True)*(1/batch_size)

        self.grada[2] = np.matmul(self.W[3].T, self.gradZ[3])
        self.gradZ[2] = np.multiply(self.grada[2], self.derivative_sigmoid(self.a[2])) 
        self.gradW[2] = np.matmul(self.gradZ[2], self.a[1].T)*(1/batch_size)
        self.gradb[2] = np.sum(self.gradZ[2], axis=1, keepdims=True)*(1/batch_size)

        self.grada[1] = np.matmul(self.W[2].T,self.gradZ[2])
        self.gradZ[1] = np.multiply(self.grada[1], self.derivative_sigmoid(self.a[1])) 
        self.gradW[1] = np.matmul(self.gradZ[1], self.a[0].T)*(1/batch_size)
        self.gradb[1] = np.sum(self.gradZ[1], axis=1, keepdims=True)*(1/batch_size)
        
        return 

            
    def update_weight(self, epoch):

        # #Adaptive LR: set step size
        # if epoch % self.steplr_step == 0:
        #     self.lr = self.lr * self.lr
        
        #momentum:
        beta = 0.9
        if self.linear == True:
            for i in range(1,4):
                #momentum
                self.vt[i] = beta * self.vt[i] - self.lr*self.gradW[i] 
                self.W[i] = self.W[i] + self.vt[i]
                self.vt2[i] = beta * self.vt2[i] - self.lr * self.gradb[i]
                self.b[i] = self.b[i] + self.vt2[i]            
        elif self.XOR == True:
            for i in range(1,4):
                #adagrad
                n = np.sum(self.gradW[i]*self.gradW[i])
                self.W[i] = self.W[i] - self.lr * (1/np.sqrt(n+self.eps))*self.gradW[i]
                n2 = np.sum(self.gradb[i]*self.gradb[i])
                self.b[i] = self.b[i] - self.lr * (1/np.sqrt(n2+self.eps))*self.gradb[i]
                     
        # for i in range(1,4):
        #     #momentum
        #     self.vt[i] = beta * self.vt[i] - self.lr*self.gradW[i] 
        #     self.W[i] = self.W[i] + self.vt[i]
        #     self.vt2[i] = beta * self.vt2[i] - self.lr * self.gradb[i]
        #     self.b[i] = self.b[i] + self.vt2[i]   
        #     #adagrad        
        #     n = np.sum(self.gradW[i]*self.gradW[i])
        #     self.W[i] = self.W[i] - self.lr * (1/np.sqrt(n+self.eps))*self.gradW[i]
        #     n2 = np.sum(self.gradb[i]*self.gradb[i])
        #     self.b[i] = self.b[i] - self.lr * (1/np.sqrt(n2+self.eps))*self.gradb[i]
            

            #original(without optimizer):
            # self.W[i] -= self.lr * self.gradW[i] 
            # self.b[i] -= self.lr * self.gradb[i]
        return
    

        

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

def generate_XOR_easy():
    inputs = []
    labels = []
    for i in range(11):
        inputs.append([0.1*i, 0.1*i])
        labels.append(0)
        if 0.1*i == 0.5:
            continue
        inputs.append([0.1*i, 1-0.1*i])
        labels.append(1)
    return np.array(inputs), np.array(labels).reshape(21,1)

def show_result(x, y, pred_y):
    plt.subplot(1,2,1)
    plt.title('Ground Truth', fontsize=18)
    for i in range(x.shape[1]):
        if y[i] == 0:
            plt.plot(x[0][i], x[1][i], 'ro')
        else:
            plt.plot(x[0][i], x[1][i], 'bo')
   
    plt.subplot(1,2,2)
    plt.title('Predict Result', fontsize=18)
    for i in range(x.shape[1]):
        if pred_y[i] == 0:
            plt.plot(x[0][i], x[1][i], 'ro')
        else:
            plt.plot(x[0][i], x[1][i], 'bo')
    plt.show()
    return 

def show_learning_curve(error):
    plt.plot(np.sqrt(error), "r-+", linewidth=0.3, label="train")
    plt.legend(loc="upper right", fontsize=14)
    plt.xlabel("Epochs", fontsize=14)
    plt.ylabel("Loss", fontsize=14)
    plt.show()
    return 

def main():

    linear_threshold = 0.015
    xor_threashold = 0.001
    early_stopping_count = 0
    linear_train_error = []
    xor_train_error = []
    
    
    
    #Linear:
    
    x, y = generate_linear()
    x = x.T
    y = y.T
   
    model = Net()
    model.linear = True
    model.XOR = False
    
    print("\n---Linear Training Result---")
    for i in range(model.epochs):            
        y_pred = model.forward(x, y)
        loss = model.calculate_loss(y, y_pred)
        linear_train_error.append(loss)
        model.backward(y, y_pred)
        model.update_weight(i)

        if i % model.output_loss_interval == 0:
            acc=(1.-np.sum(np.abs(y-np.round(y_pred)))/y.shape[1])*100
            print(f"Epochs {i}: loss={loss} accuracy={acc}%")
        
        if abs(loss) <= linear_threshold:
            early_stopping_count += 1
            if early_stopping_count == 3:
                print(f"\n\n---Early Stopping at epoch: {i} of loss: {loss} acc:{acc}---\n\n")
                early_stopping_count = 0
                break

    show_result(x, y[0], np.round(y_pred[0]))
    show_learning_curve(linear_train_error)
    print("\n---Linear Testing Result---")
    y_pred_linear_test = model.forward(x,y)
    y_pred_linear_loss = model.calculate_loss(y, y_pred_linear_test)
    acc=(1.-np.sum(np.abs(y-np.round(y_pred_linear_test)))/y.shape[1])*100
    print(f"loss={y_pred_linear_loss} accuracy={acc}%")




    #XOR
    x,y = generate_XOR_easy()
    x, y= x.T, y.T
    model = Net()
    model.linear = False
    model.XOR = True
    
    
    #train
    print("\n---XOR Training Results---")
    for i in range(model.epochs):
        y_pred = model.forward(x, y)
        loss = model.calculate_loss(y, y_pred)
        xor_train_error.append(loss)
        model.backward(y, y_pred)
        model.update_weight(i)

        if i % model.output_loss_interval == 0:
            acc=(1.-np.sum(np.abs(y-np.round(y_pred)))/y.shape[1])*100
            print(f"Epochs {i}: loss={loss} accuracy={acc}%")

        #early stopping
        if abs(loss) <= xor_threashold:
            early_stopping_count += 1
            if early_stopping_count == 3:
                print(f"\n\n---Early Stopping at epoch: {i} of loss: {loss} acc:{acc}---\n\n")
                break
    show_result(x, y[0], np.round(y_pred[0]))
    show_learning_curve(xor_train_error)

    #test
    print("\n---XOR Testing Results---")
    y_xor_test_pred = model.forward(x,y)
    xor_test_loss = model.calculate_loss(y, y_xor_test_pred)
    acc=(1.-np.sum(np.abs(y-np.round(y_xor_test_pred)))/y.shape[1])*100
    print(f"loss={xor_test_loss} accuracy={acc}%")

    
    


if __name__ == '__main__':
    main()