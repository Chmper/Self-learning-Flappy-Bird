import numpy as np
import random

''' Obczaic czy metody z generacji (crossover, mutate, copy) powinny byc
    tutaj. Niby spoko bo mozna by uzywac jednej sieci do wielu projektow '''

class Neural_Network:
    def __init__(self, inputs, hidden):
        self.w_1 = 2*np.random.rand(inputs, hidden)-1
        self.w_2 = 2*np.random.rand(1,hidden).T-1
        self.inputs = inputs
        self.hidden = hidden

    def sigm(self, x):
        return 1/(1+np.exp(-x))

    def predict(self, inputs):
        first_layer = np.dot(inputs, self.w_1)
        first_layer = self.sigm(first_layer)

        return self.sigm(np.dot(first_layer, self.w_2))

    def mutate(self):
        self.w_1 += (2*np.random.rand(self.inputs, self.hidden)-1)/10

    def crossover(self, neural2):

        off = Neural_Network(self.inputs, self.hidden)
        point = random.randint(1,4)

        off.w_1[0] = np.concatenate((self.w_1[0][:point], neural2.w_1[0][point:]))
        off.w_1[1] = np.concatenate((self.w_1[1][:point], neural2.w_1[1][point:]))

        off.w_2[0] = np.concatenate((self.w_2[0][:point], neural2.w_2[0][point:]))
        off.w_2[1] = np.concatenate((self.w_2[1][:point], neural2.w_2[1][point:]))

        return off
