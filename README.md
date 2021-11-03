# Deep Learning Side Projects


#### Lab 1: Backpropagation from scratch (without any framework such as PyTorch)
In this task, a simple neural network should be implemented, with forwarding pass and backpropagation using two hidden layers. Notice that only Numpy and python standard libraries can be used. Any other frameworks are not allowed. 
![image](https://user-images.githubusercontent.com/36917138/140075490-aa92ab8b-3501-454f-bf8a-8efc8bf03dbb.png)
Input:
![image](https://user-images.githubusercontent.com/36917138/140075617-68185957-6d97-414f-a717-eafcadc4b162.png)

#### Lab 2: Temporal Difference Learning (2048 Game)
In this task, a temporal difference learning (TD) algorithm will be implemented by solving the 2048 game using an n-tuple network. 
![image](https://user-images.githubusercontent.com/36917138/140075994-ac10b2a1-bdca-4296-b60a-37d6ea305a42.png)

#### Lab 3: EEG Classification
In this task, a simple EEG Classification model will be implemented. There are in total two ways to construct the model, which are EEGNet, DeepConvNet. Dataset in use is BCI competition dataset. 
Additionally, different kinds of activation functions are tried to be included in the task, which are ReLU, LeakyReLU, ELU.
Input:
![image](https://user-images.githubusercontent.com/36917138/140076512-b1b79415-adc3-4800-b173-22f9940f75be.png)
EEGNet Structure:
![image](https://user-images.githubusercontent.com/36917138/140076568-f3f929d2-6ad5-4eff-92e4-819bbd50abc1.png)

#### Lab 4: Diabetic Retinopathy Detection
In this task, an analysis on diabetic retinopathy model  is carried out. The task includes three different modules: constructing own DataLoader through PyTorch framework, classifying diabetic retinopathy grading via ResNet architecture, and calculating the confusion matrix for performance evaluation. 
Input:
![image](https://user-images.githubusercontent.com/36917138/140077113-d26420b4-1607-4dea-844c-0bcef6afcf58.png)

#### Lab 5: Conditional Sequence-to-Sequence VAE
In this task, a conditional seq2seq VAE for English tense convrsion and generation model is implemented. The model is able to do English tense conversion and text generation, such as when we input the input word “access” with the tense (condition) “simple present” to the encoder, it has to generate a latent vector z. Then we take z with the tense “present progressive” as the input for the decoder, it’s expected to output word “accessing”. 
The structure:
![image](https://user-images.githubusercontent.com/36917138/140077656-fe61e4f1-b850-4a65-afb2-cc015fb35f9e.png)

#### Lab 6: DQN and DDPG
In this task, two deep reinforcement algorithms are implemented, which are 
(1) Solve LunarLander-v2 using deep Q-network (DQN)
(2) Solve LunarLanderContinuous-v2 using deep deterministic policy gradient (DDPG)
![image](https://user-images.githubusercontent.com/36917138/140077972-534dbc50-4295-4757-b5af-66f245a36b3c.png)

#### Lab 7: Let’s play GANs with Flows and Friends
In this task, two models: generative adversarial network (GAN) and a normalizing flow network are implemented to solve the following two problems. 

For the first problem, both of the generative models should be conditional. They should be able to generate synthetic object images with multi-label conditions. For example, given “red cube” and “blue cylinder”, the model should generate the synthetic images with red cube and blue cylinder. After generation, you need to input generated images to a pre-trained classifier for evaluation.
![image](https://user-images.githubusercontent.com/36917138/140078807-a114027a-e31c-469b-9796-6982b5f2324f.png)

The second problem is human face generation.The implemented normalizing flow model should be able to generate human faces. The task includes:
(1) Conditional face generation
(2) Linear Interpolation
![image](https://user-images.githubusercontent.com/36917138/140078885-0c13a01e-427c-4384-a0c8-83999362be21.png)
(3) Attribute manipulation
![image](https://user-images.githubusercontent.com/36917138/140078911-bbc335cf-90b3-41dc-bb04-0a2ba752b4c8.png)
