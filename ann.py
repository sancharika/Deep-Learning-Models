#neural network

import numpy
import scipy.special

input_nodes = 3
hidden_nodes = 3
output_nodes = 3

wih = numpy.random.normal(0.0,pow(input_nodes,-0.5),(hidden_nodes,input_nodes))
who  = numpy.random.normal(0.0,pow(hidden_nodes,-0.5),(output_nodes,hidden_nodes))

lr = 0.3
activation_function = lambda x : scipy.special.expit(x)

inputs_list = [0.2,0.3,0.4]
targets_list = [0.3,0.2,0.1]

inputs = numpy.array(inputs_list,ndmin=2).T
targets = numpy.array(targets_list,ndmin = 2).T

hidden_inputs = numpy.dot(wih,inputs)
hidden_outputs = activation_function(hidden_inputs)

final_inputs = numpy.dot(who,hidden_outputs)

final_outputs = activation_function(final_inputs)

output_errors = targets - final_outputs

hidden_errors = numpy.dot(who.T,output_errors)

who += lr * numpy.dot((output_errors * final_outputs * (1.0 - final_outputs)), numpy.transpose(hidden_outputs))

wih += lr * numpy.dot((hidden_errors * hidden_outputs * (1.0 - hidden_outputs)), numpy.transpose(inputs))

inputs = numpy.array(inputs_list,ndmin=2).T
hidden_inputs = numpy.dot(wih,inputs)
hidden_outputs = activation_function(hidden_inputs)
final_inputs = numpy.dot(who,hidden_outputs)
final_outputs = activation_function(final_inputs)

print("Final outputs")
print(final_outputs)
