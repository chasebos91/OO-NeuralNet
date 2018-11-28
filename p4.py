import numpy as np
import pandas as pd
from sklearn import preprocessing
import random

#CONSTS
SEPAL_LENGTH = 0
SEPAL_WIDTH = 1
PETAL_LENGTH = 2
PETAL_WIDTH = 3
CLASS = 4
SETOSA = 0
VERSICOLOUR = 1
VIRGINICA = 2
INPUT_LAYER = 0
HIDDEN_LAYER = 1
OUTPUT_LAYER = 2

class Neuron:

    def __init__(self, name):
        self.value = 0
        self.name = name
        self.synapse_list = []
        self.error = 0

    def set_value(self, value):
        self.value = value

    def sigmoid(self):

        return 1 / (1 + np.exp(-self.value))

    def ddx_sigmoid(self):
        return self.value * (1 - self.value)

class Synapse:

    def __init__(self, origin, dest):
        #list of nodes from which it is connected
        self.origin = origin
        # list of nodes to which it is connected
        self.dest = dest
        self.weight = random.uniform(.01,.5)
        self.delta_weight = 0


class Layer:

    def __init__(self, neuron_list, num_neurons, name):
        self.neuron_list = neuron_list
        self.num_neurons = num_neurons
        self.name = name

    def init_synapses(self, next_layer):

        for neuro_a in self.neuron_list:
            for neuro_b in next_layer.neuron_list:
                syn = Synapse(neuro_a, neuro_b)
                neuro_a.synapse_list.append(syn)

class InputLayer(Layer):



    def take_input(self, flower):
        attr = 0
        for i in self.neuron_list:
                i.set_value(flower[attr])
                attr += 1

class HiddenLayer(Layer):
# wrapper class
    pass

class OutputLayer(Layer):
# wrapper class
    pass

class Bias:

    def __init__(self):
        self.value = 1
        self.hid_list = []
        self.outp_list = []

class NeuralNetwork:

    def __init__(self, layer_list):
        self.list_of_layers = layer_list
        self.num_layers = len(self.list_of_layers)
        self.global_err = 0
        self.learning_rate = .3
        self.bias = Bias()
        self.expected_output_vals = []
        self.accuracy = float(0)
        self.connect_layers()


    def connect_layers(self):
        # connect neurons in input to hidden, hidden and output layer
        for layer in range(INPUT_LAYER, OUTPUT_LAYER):
            self.list_of_layers[layer].init_synapses(self.list_of_layers[layer+1])

        # connect bias to neurons in hidden
        for neuron in self.list_of_layers[HIDDEN_LAYER].neuron_list:
            syn = Synapse(self.bias, neuron)
            self.bias.hid_list.append(syn)

        # connect bias to neurons in output
        for neuron in self.list_of_layers[OUTPUT_LAYER].neuron_list:
            syn = Synapse(self.bias, neuron)
            self.bias.outp_list.append(syn)

    # sets the list of expected output values of the output neurons
    def set_expected_output(self, flower):

        for i in range(0,self.list_of_layers[OUTPUT_LAYER].num_neurons):
            if i == flower[CLASS]:
                self.expected_output_vals.append(.98)
            else: self.expected_output_vals.append(.01)

    # checks if neural net guessed right
    def is_correct(self, flower):

        maxval = 0.0
        max_neuron = None

        for output_neuron in range(0,self.list_of_layers[OUTPUT_LAYER].num_neurons):

            if self.list_of_layers[OUTPUT_LAYER].neuron_list[output_neuron].value > maxval:
                maxval = self.list_of_layers[OUTPUT_LAYER].neuron_list[output_neuron].value
                max_neuron = output_neuron

        if flower[CLASS] == max_neuron: return 1

        return 0

    # calculates the global error of the neural net
    def calc_global_err(self):
        i = 0
        self.global_err = 0
        for node in self.list_of_layers[OUTPUT_LAYER].neuron_list:
            self.global_err += ((self.expected_output_vals[i] - node.value)**2) / 2
            i += 1

    # feeds the input from flower into the input layer, then props forward
    def forward_pass(self, flower):

        self.list_of_layers[0].take_input(flower)
        self.set_expected_output(flower)

        # feed bias forward
        for node in self.bias.hid_list:
            node.dest.value += node.origin.value * node.weight

        for node in self.bias.outp_list:
            node.dest.value += node.origin.value * node.weight

        for neuron in self.list_of_layers[INPUT_LAYER].neuron_list:
            for syn in neuron.synapse_list:
                syn.dest.value += syn.origin.value * syn.weight

        for neuron in self.list_of_layers[HIDDEN_LAYER].neuron_list:
            neuron.value = neuron.sigmoid()

        for neuron in self.list_of_layers[HIDDEN_LAYER].neuron_list:
            for syn in neuron.synapse_list:
                syn.dest.value += syn.origin.value * syn.weight

        for neuron in self.list_of_layers[OUTPUT_LAYER].neuron_list:
            neuron.value = neuron.sigmoid()

        # testing
        print("flower:")
        for atr in flower:
            print(atr)

        print("Before back pass")
        for i in self.list_of_layers:
            for j in i.neuron_list:
                for syn in j.synapse_list:
                    print(syn.origin.name, syn.dest.name, syn.weight)
        print("bias:")
        for node in self.bias.hid_list:
            print(node.dest.name, node.weight)

        for node in self.bias.outp_list:
            print(node.dest.name, node.weight)


    def clear_vals(self):
        for layer in self.list_of_layers:
            for neuron in layer.neuron_list:
                neuron.value = 0

    def back_pass(self):

        # Step 1 - calculate total error
        target = 0

        for outp_neuron in self.list_of_layers[OUTPUT_LAYER].neuron_list:
            outp_err = (self.expected_output_vals[target] - outp_neuron.value) \
                        * outp_neuron.ddx_sigmoid()
            target += 1
        # Step 2: calc contribution to error from hidden neuron
            for hid_neuron in self.list_of_layers[HIDDEN_LAYER].neuron_list:
                for syn in hid_neuron.synapse_list:
                    if syn.dest == outp_neuron:
                        hid_neuron.error = syn.weight * outp_err
                        outp_neuron.error += hid_neuron.error

            # summing the error from weight between bias->outp_neuron
            for bias_syn in self.bias.outp_list:
                if bias_syn.dest == outp_neuron:
                    outp_neuron.error += bias_syn.weight * outp_err

            outp_neuron.error = outp_neuron.error * outp_neuron.ddx_sigmoid()

            # Step 3 rate of change
            # this value will be used to change the weights that are
            # connected to this particular output neuron
            outp_delta = outp_neuron.error * outp_err * self.learning_rate

            for hid_neuron in self.list_of_layers[HIDDEN_LAYER].neuron_list:
                for syn in hid_neuron.synapse_list:
                    if syn.dest == outp_neuron:
                        syn.weight += outp_delta

            # save delta of bias weight connected to output neuron
            for syn in self.bias.outp_list:
                if syn.dest == outp_neuron:
                    syn.delta_weight = outp_delta
                    syn.weight += syn.delta_weight

            # i_neuron->h_neuron delta weights
            for h_neuron in self.list_of_layers[HIDDEN_LAYER].neuron_list:
                for i_neuron in self.list_of_layers[INPUT_LAYER].neuron_list:
                    for syn in i_neuron.synapse_list:
                        if syn.dest == h_neuron:
                            syn.delta_weight = h_neuron.error * \
                                                h_neuron.sigmoid() * \
                                                self.learning_rate

            # for bias -> h_neuron
                for syn in self.bias.hid_list:
                    syn.delta_weight = h_neuron.error * \
                                           h_neuron.sigmoid() * \
                                           self.learning_rate
                    syn.weight += syn.delta_weight

            # update weights for i_neuron -> h_neuron
            for neuron in self.list_of_layers[INPUT_LAYER].neuron_list:
                for syn in neuron.synapse_list:
                    syn.weight += syn.delta_weight



        print("After")
        for i in self.list_of_layers:
            for j in i.neuron_list:
                for syn in j.synapse_list:
                    print(syn.origin.name, syn.dest.name, syn.weight)

        print("bias:")
        for node in self.bias.hid_list:
            print(node.dest.name, node.weight)

        for node in self.bias.outp_list:
            print(node.dest.name, node.weight)

        self.clear_vals()
        self.expected_output_vals.clear()

    def train(self, flower_list, runs):

        iters = 0
        correct_answer = 0
        incorrect_answer = 0

        while iters < runs:
            for flower in flower_list:
                self.forward_pass(flower)
                if self.is_correct(flower) == 1:
                    print("is correct: ", self.is_correct(flower))
                    correct_answer += 1
                else:
                    print("is correct: 0")
                    incorrect_answer += 1

                self.calc_global_err()
                self.back_pass()

            iters += 1

        # for flower in flower_list:
        #    print("is correct: ", self.is_correct(flower))
        #    correct_answer += self.is_correct(flower)
        #    if self.is_correct(flower) == 0:
        #        incorrect_answer += 1

        self.set_error_rate(correct_answer, iters)
        print(correct_answer)
        print(incorrect_answer)
        print("ACCURACY:", self.accuracy)
        print("iterations:", iters)


    def set_error_rate(self, correct_answer, iters):

        self.accuracy = correct_answer / iters

    def validation(self, flower_list):

        iters = 0
        correct_answer = 0
        incorrect_answer = 0

        for flower in flower_list:
            self.forward_pass(flower)
            if self.is_correct(flower) == 1:
                #print("is correct: ", self.is_correct(flower))
                correct_answer += 1
            else:
                #print("is correct: 0")
                incorrect_answer += 1
            iters += 1

        accuracy = correct_answer/iters
        print("Accuracy in Validation:", accuracy)

        if accuracy > self.accuracy:
            print("NN is overfitted.")


    def testing(self, flower):
        iters = 0
        correct_answer = 0
        incorrect_answer = 0

        for flower in flower_list:
            self.forward_pass(flower)
            if self.is_correct(flower) == 1:
                correct_answer += 1
            else:
                incorrect_answer += 1
            iters += 1

        accuracy = correct_answer / iters
        print("Accuracy in Testing:", accuracy)

    def guess(self, flower):

        self.forward_pass(flower)
        if self.is_correct(flower) == 1:
            print("CORRECT!")
        else:
            print("BAD NN BAD.")

# MAIN FUNCTIONS

def add_to_list(list, **kwargs):
    for i in kwargs:
        list.append(kwargs[i])
    return list

def categorical_data(flower):
    if flower[CLASS] == "Iris-setosa":
        flower[CLASS] = SETOSA
    elif flower[CLASS] == "Iris-versicolor":
        flower[CLASS] = VERSICOLOUR
    else:
        flower[CLASS] = VIRGINICA

def take_user_input():

    s_length = input("Enter sepal length:")
    s_width =  input("Enter sepal width:")
    p_length = input("Enter petal length:")
    p_width = input("Enter petal width:")
    f_class = input("Enter flower class (0 for setosa, 1 for versicolour, and 2 for virginica).")

    flower = [s_length, s_width, p_length, p_width, f_class]

    return flower

def normalize(min, max, actual):

    new = (actual - min)/(max - min)
    return new

######## MAIN

# file maintenance and data normalization
#---------------------------------------
flower_file = pd.read_csv("PS4 - Iris data.csv", header = None, names = ["S-Width", "S-Length", "P-Width", "P-Length", "CLASS"])
flower_file = flower_file[:-1]

s1_max = flower_file["S-Width"].max()
s1_min = flower_file["S-Width"].min()

s2_max = flower_file["S-Length"].max()
s2_min = flower_file["S-Length"].min()

p1_max = flower_file["P-Width"].max()
p1_min = flower_file["P-Width"].min()


p2_max = flower_file["P-Length"].max()
p2_min = flower_file["P-Length"].min()

s_width = flower_file[["S-Width"]].values.astype(float)
min_max_scale1 = preprocessing.MinMaxScaler()
s_width_scaled = min_max_scale1.fit_transform(s_width)
normalized = pd.DataFrame(s_width_scaled)

s_length = flower_file[["S-Length"]].values.astype(float)
min_max_scale2 = preprocessing.MinMaxScaler()
s_length_scaled = min_max_scale2.fit_transform(s_length)
normalized2 = pd.DataFrame(s_length_scaled)

p_width = flower_file[["P-Width"]].values.astype(float)
min_max_scale3 = preprocessing.MinMaxScaler()
p_width_scaled = min_max_scale3.fit_transform(p_width)
normalized3 = pd.DataFrame(p_width_scaled)

p_length = flower_file[["P-Length"]].values.astype(float)
min_max_scale4 = preprocessing.MinMaxScaler()
p_length_scaled = min_max_scale4.fit_transform(p_length)
normalized4 = pd.DataFrame(p_length_scaled)

normalized["S-Length"] = normalized2
normalized["P-Width"] = normalized3
normalized["P-Length"] = normalized4
normalized["CLASS"] = flower_file["CLASS"]
normalized.columns = ["S-Width","S-Length", "P-Width","P-Length","CLASS"]
flower_list = normalized.values.tolist()
for flower in flower_list:
    if flower[CLASS] == "Iris-setosa":
        flower[CLASS] = SETOSA
    elif flower[CLASS] == "Iris-versicolor":
        flower[CLASS] = VERSICOLOUR
    else: flower[CLASS] = VIRGINICA

#shuffle the list and make training/valid/test sets
random.shuffle(flower_list)

training_set = []
validation = []
test = []

for i in range(0,90):
    training_set.append(flower_list[i])

for j in range(90,120):
    validation.append(flower_list[j])

for k in range(120,150):
    test.append(flower_list[k])

#-----------------------------------------

# Build/Initialize NN

list1 = []
input1 = Neuron("i1")
input2 = Neuron("i2")
input3 = Neuron("i3")
input4 = Neuron("i4")

list1 = add_to_list(list1, a = input1, b = input2, c = input3, d = input4)
in_layer = InputLayer(list1, len(list1), "in")

list2 = []
hid1 = Neuron("h1")
hid2 = Neuron("h2")
hid3 = Neuron("h3")
list2 = add_to_list(list2, a = hid1, b = hid2, c = hid3)
hid_layer = HiddenLayer(list2, len(list2), "hid")

list3 = []
out1 = Neuron("o1 - setosa")
out2 = Neuron("o2 - veris")
out3 = Neuron("o3 - virginica")
list3 = add_to_list(list3, a = out1, b = out2, c = out3)
out_layer = OutputLayer(list3, len(list3), "out")

layer_list = []
layer_list = add_to_list(layer_list, a = in_layer, b = hid_layer, c = out_layer)
NN = NeuralNetwork(layer_list)

NN.train(training_set, 1)


NN.validation(validation)
NN.testing(test)
"""
flower = take_user_input()

# normalize user input
flower[0] = normalize(s1_min, s1_max, float(flower[0]))
flower[1] = normalize(s2_min, s2_max, float(flower[1]))
flower[2] = normalize(p1_min, p1_max, float(flower[2]))
flower[3] = normalize(p2_min, p2_max, float(flower[4]))

flower_list.clear()
flower_list.append(flower)

NN.guess(flower)

"""



