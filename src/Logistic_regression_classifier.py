import numpy as np
import pandas as pd

class Logistic_regression_classifier():

    def __init__(self, vocabulary, training_set):
        
        self.vocabulary = vocabulary
        self.training_set = training_set

        self.y_real = self.extract_classes()

        self.X = self.create_BOW()

        n_features = self.X.shape[1]
        self.weights, self.bias = self.initialize_parameters(n_features)

        # Hyperparameters
        learning_rate = 0.5
        num_iter = 10000

        # Train model
        self.weights, self.bias = self.gradient_descent(learning_rate, num_iter)


    def extract_classes(self):

        y_real = []

        for i in range(len(self.training_set)):

            if self.training_set[i][-1] == "spam":

                y_real.append(1)

            else:

                y_real.append(0)

            self.training_set[i].pop()

        y_real = np.array(y_real).reshape((len(y_real),1))

        return y_real


    def create_BOW(self):

        vocabulary_size = len(self.vocabulary)
        training_set_cardinality = len(self.training_set)
        
        shape = (training_set_cardinality, vocabulary_size)

        X = pd.DataFrame(np.zeros(shape), columns=self.vocabulary, index=[f"Message {i}" for i in range(training_set_cardinality)])

        for word in self.vocabulary:

            for i in range(len(self.training_set)):

                if word in self.training_set[i]:

                    X.at[f"Message {i}", word] += 1

        return X.to_numpy()


    def sigmoid(self, z):

        return 1 / (1 + np.exp(-z))

    
    def initialize_parameters(self, n_features):
        # Initialize the coefficients (weights) to 0
        self.weights = np.zeros(n_features)
        self.weights = self.weights.reshape((n_features,1))
        self.bias = 0
        return self.weights, self.bias
    

    def predict(self, X):
        
        z = np.dot(X, self.weights) + self.bias # Logit vector

        y_predicted = self.sigmoid(z) #=ŷ

        return y_predicted
    

    def compute_loss_and_gradient(self):

        m = self.X.shape[0]  # Number of messages
        y_predicted = self.predict(self.X)
        
        # Compute log-loss
        a = np.multiply(self.y_real, np.log(y_predicted))
        b = np.multiply((1 - self.y_real), np.log(1 - y_predicted))
        loss = - (1 / m) * np.sum(a + b) # AKA J
        
        # Gradients of weight and bias   
        dw = (1 / m) * np.dot(self.X.T, (y_predicted - self.y_real))
        db = (1 / m) * np.sum(y_predicted - self.y_real)
        
        return loss, dw, db
    

    def gradient_descent(self, learning_rate, num_iter):
        
        for i in range(num_iter):
            #Compute loss and gradient
            loss, dw, db = self.compute_loss_and_gradient()

            # Update parameters
            self.weights = self.weights - learning_rate * dw
            self.bias = self.bias - learning_rate * db
        
        return self.weights, self.bias
    





