
# coding: utf-8

# In[81]:

# importing numpy for matrix multiplication 
import numpy

# importing scipy to use sigmoid for the activation function
import scipy.special

# importing matplotlib to be able to view the number images
import matplotlib.pyplot

# this just makes sure that the images appear inline rather than in a seperate window
get_ipython().magic('matplotlib inline')


# creating a new class called neural_network. This is like a template for any networks made in future, with certain
# associated properties and functions. NB: A function associated with an object is called a method.
class neural_network:
    
    # to initialise a new neural network, provide the number of input, hidden, and output nodes, and the learning rate. 
    # self just refers to the instance created
    def __init__(self, num_inputs, num_hiddens, num_outputs, learning_rate):
        
        # setting the properties of the instance created to the values provided
        self.num_inputs = num_inputs
        self.num_hiddens = num_hiddens
        self.num_outputs = num_outputs
        self.learning_rate = learning_rate
        self.activation_function = lambda x: scipy.special.expit(x)
        
        # creating matrices of weights: inputs to hiddens, and hiddens to outputs.
        # initial random weights are distributed centered around zero with a standard
        # deviation relative to the number of incoming links to the node. In this 
        # case 1/√(incoming links).
        self.hidden_input_weights = numpy.random.normal(0.0, pow(self.num_hiddens, -0.5), (self.num_hiddens, self.num_inputs))
        self.hidden_output_weights = numpy.random.normal(0.0, pow(self.num_outputs, -0.5), (self.num_outputs, self.num_hiddens))
    
    # creating a function to train the network taking a list of inputs and a list of correct target outputs
    def train(self, input_list, target_list):
        
        # converting lists to arrays so we can use them for matrix multiplication.
        # They are transposed, but why they must be, I do not know. 
        target = numpy.array(target_list, ndmin=2).T
        inputs = numpy.array(input_list, ndmin=2).T
        
        # dot product of the inputs and the hidden input weights provides the input to the hidden layer
        hidden_input = numpy.dot(self.hidden_input_weights, inputs)
        
        # and the output of the hidden layer is just the activation function (using sigmoid here) applied to the
        # input to the hidden layer
        hidden_output = self.activation_function(hidden_input)
        
        # input to the final layer is just the dot product of the output of the hidden layer and the hidden
        # output weights
        final_input = numpy.dot(self.hidden_output_weights, hidden_output)
        
        # one more activation function applied to the final layer input provides the final output
        final_output = self.activation_function(final_input)
        
        # and the output error is easy: just the difference between the target outputs and the actual output
        output_error = target - final_output
        
        # then to get the error in the hidden layer, pass the output error back along the weights to the hidden layer
        hidden_error = numpy.dot(self.hidden_output_weights.T, output_error)
        
        # how to adjust the weights according to the error? This is the tricky bit. I know what it's for but I'm not
        # sure I follow the WHY of the maths here. 
        hidden_output_weight_change = self.learning_rate * numpy.dot((output_error * final_output * (1.0 - final_output)), numpy.transpose(hidden_output)) 
        hidden_input_weight_change = self.learning_rate * numpy.dot((hidden_error * hidden_output * (1.0 - hidden_output)), numpy.transpose(inputs))
        self.hidden_output_weights += hidden_output_weight_change
        self.hidden_input_weights += hidden_input_weight_change
    
    # gives a list of inputs and hopes the network knows
    def query(self, input_list):
        
        #converting that list input to an array. Transposed. I will figure out why it needs to be so tomorrow.
        inputs = numpy.array(input_list, ndmin=2).T
        
        # and here just taking that input through the nodes, applying the weights learned through training.
        hidden_input = numpy.dot(self.hidden_input_weights, inputs)
        hidden_output = self.activation_function(hidden_input)
        
        final_input = numpy.dot(self.hidden_output_weights, hidden_output)
        final_output = self.activation_function(final_input)
        
        return final_output
    
    
    
# creating a new instance of a neural network with 784 input nodes (the no. of pixels in each handwritten number
# image), 100 hidden nodes (just a good number), and 10 output nodes (for each number 0 - 9 that we could
# identify) and a learning rate of 0.3

num_of_inputs = 784
num_of_hiddens = 100
num_of_outputs = 10
learning_rate = 0.3

nn = neural_network(num_of_inputs,num_of_hiddens,num_of_outputs,learning_rate)
    
# opening the file cintaining training data
data_in = open("mnist_train.csv", 'r')

# reading in all the lines from that file
data = data_in.readlines()

# don't forget to close the file!
data_in.close()

# now train the neural network!
# for each line (there is one img per line, a string of numbers representing pixel colour values seperated by commas.)
for line in data:

    # splitting this string at the commas provides a list of ints
    img_data = line.split(',')

    # the first int in the list is the number that the image is of. So when reading in the image it should be ignored.
    # to avoid problems with zero valued input killing link weights, rescale the data so each value is between 0.01 and 1
    scaled_img_array = (numpy.asfarray(img_data[1:]) / 255.0 * 0.99) + 0.01
  
    # scaled_img_array is now ready to be used as input to train the network.
    
    # What should the target output be? It could be any number from 0 - 9. We want it to be the number that the input
    # values represent. We have 10 outputs representing the 10 possible numbers, and we want only the correct one to
    # fire. This could be represented as 1 and unwanted outputs as 0, but since our activation function cannot output 0
    # or 1, only numbers within that range, it's better to use 0.01 and 0.99.

    # create an array of length num_of_inputs (in this case 10) with each value set to 0.01
    target = numpy.zeros(num_of_outputs) + 0.01

    # using that first value in img_data (converted to int) to set correct item in target array to 0.99
    target[int(img_data[0])] = 0.99
    
    nn.train(scaled_img_array,target)
    
# The network should now be trained and ready to go! Let us test it.
test_file = open("mnist_test.csv", 'r')
test_data = test_file.readlines()
test_file.close()

# keep track of how accurate the network is
results = []

# testing the first image
for line in test_data:

    test_img = line.split(',')
    scaled_test_img_array = (numpy.asfarray(test_img[1:]) / 255.0 * 0.99) + 0.01

    target = line[0]

    array_output = nn.query(scaled_test_img_array)
    output = numpy.argmax(array_output)
    
    if (target == str(output)):
        results.append(1)
    else:
        results.append(0)
        

performance = sum(results) / len(results)

print("performance: " + str(performance))


# In[ ]:



