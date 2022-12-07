# Image Processing Sample Work 
A new repository for a previous task on image and language recognition 
:computer: :computer:

Please read the description below before checking out the report PDFs as well as the Python scripts for each processing task. 

**Description of tasks**

1. Image Recognition (Japanese character recognition)

Implementing networks to recognise handwritten Hiragana symbols. The dataset to be used is Kuzushiji-MNIST or KMNIST for short. Significant changes occurred to the language when Japan reformed their education system in 1868, and the majority of Japanese today cannot read texts published over 150 years ago. The dataset contains 10 Hiragana characters with 7000 samples per class. This is the dataset we will be using. The steps implemented for part 1 are as follows: 

**PART 1**

Step 1 - Implement a model NetLin which computes a linear function of the pixels in the image, followed by log softmax. In Python on your computer, run the code by typing:

python3 kuzu_main.py --net lin

Copy the final accuracy and confusion matrix into your report. The final accuracy should be around 70%. Note that the rows of the confusion matrix indicate the target character, while the columns indicate the one chosen by the network. (0="o", 1="ki", 2="su", 3="tsu", 4="na", 5="ha", 6="ma", 7="ya", 8="re", 9="wo"). More examples of each character can be found here.

Step 2 - Implement a fully connected 2-layer network NetFull (i.e. one hidden layer, plus the output layer), using tanh at the hidden layer and log softmax at the output layer. Run the code by typing:

python3 kuzu_main.py --net full

Try different values (multiples of 10) for the number of hidden nodes and try to determine a value that achieves high accuracy (at least 84%) on the test set. Copy the final accuracy and confusion matrix into your report.

Step 3 - Implement a convolutional network called NetConv, with two convolutional layers plus one fully connected layer, all using relu activation function, followed by the output layer, using log softmax. You are free to choose for yourself the number and size of the filters, metaparameter values (learning rate and momentum), and whether to use max pooling or a fully convolutional architecture. Run the code by typing:

python3 kuzu_main.py --net conv.

Your network should consistently achieve at least 93% accuracy on the test set after 10 training epochs. Copy the final accuracy and confusion matrix into your report.

Step 4 - Discuss what you have learned from this exercise, including the following points:

the relative accuracy of the three models, the confusion matrix for each model: which characters are most likely to be mistaken for which other characters, and why? experiment with other architectures and/or metaparameters for this dataset, and report on your results; the aim of this exercise is not only to achieve high accuracy but also to understand the effect of different choices on the final accuracy. Part 2 - Twin spirals Two Spirals Data from (Lang & Witbrock, 1988)

**Part 2**

 Part 2, you will be training on the famous Two Spirals Problem (Lang and Witbrock, 1988). The supplied code spiral_main.py loads the training data from spirals.csv, applies the specified model and produces a graph of the resulting function, along with the data. For this task there is no test set as such, but we instead judge the generalization by plotting the function computed by the network and making a visual assessment.

This part of the assessment is comprised of six steps:

Step 1 - Provide code for a Pytorch Module called PolarNet which operates as follows: First, the input (x,y) is converted to polar co-ordinates (r,a) with r=sqrt(xx + yy), a=atan2(y,x). Next, (r,a) is fed into a fully connected neural network with one hidden layer using tanh activation, followed by a single output using sigmoid activation. The conversion to polar coordinates should be included in your forward() method, so that the Module performs the entire task of conversion followed by network layers.

Step 2 - In Python on your computer, run the code by typing:

python3 spiral_main.py --net polar --hid 10.

Try to find the minimum number of hidden nodes required so that this PolarNet learns to correctly classify all of the training data within 20000 epochs, on almost all runs. The graph_output() method will generate a picture of the function computed by your PolarNet called polar_out.png, which you should include in your report.

Step 3 - Provide code for a Pytorch Module called RawNet which operates on the raw input (x,y) without converting to polar coordinates. Your network should consist of two fully connected hidden layers with tanh activation, plus the output layer, with sigmoid activation. The two hidden layers should each have the same number of hidden nodes, determined by the parameter num_hid. Note that this network differs from the one depicted in slide 10 of lecture slides 3a on Hidden Unit Dynamics, in two ways: Firstly, only two hidden nodes are shown in the diagram, but a larger number of hidden nodes are needed for the task. Secondly, our network will not have shortcut connections; although these appeared to be necessary in the original 1988 paper using SGD, it now appears that the task can be learned without shortcut connections, with the help of the Adam optimizer.

Step 4 - In Python on your computer, run the code by typing:

python3 spiral_main.py --net raw

Try to choose a value for the number of hidden nodes (--hid) and the size of the initial weights (--init) such that this RawNet learns to correctly classify all of the training data within 20000 epochs, on almost all runs. Include in your report the number of hidden nodes, and the values of any other metaparameters. The graph_output() method will generate a picture of the function computed by your RawNet called raw_out.png, which you should include in your report.

Step 5 - Using graph_output() as a guide, write a method called graph_hidden (net, layer, node) which plots the activation (after applying the tanh function) of the hidden node with the specified number (node) in the specified layer (1 or 2). (Note: if net is of type PolarNet, graph_output() only needs to behave correctly when layer is 1). Hint: you might need to modify forward() so that the hidden unit activations are retained, i.e. replace hid1 = torch.tanh(...) with self.hid1 = torch.tanh(...). Use this code to generate plots of all the hidden nodes in PolarNet, and all the hidden nodes in both layers of RawNet, and include them in your report.

Step 6 - Discuss what you have learned from this exercise, including the following points:

The qualitative difference between the functions computed by the nodes in the hidden layer(s) in PolarNet and RawNet A brief description of how the network uses these functions to achieve the classification The effect of different values for initial weight size on the speed and success of learning for RawNet Experiment with other changes and comment on the result - for example, changing batch size from 97 to 194, using SGD instead of Adam, changing tanh to relu, adding a third hidden layer, etc. The aim is to understand how different choices impact the final result. 

**Ignore part 3 of the report, that was a harder task not many people got for the assignment**


