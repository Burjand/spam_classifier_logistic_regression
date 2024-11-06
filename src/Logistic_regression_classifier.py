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
        learning_rate = 0.01
        epochs = 1000

        # Train model
        self.weights, self.bias = self.gradient_descent(learning_rate, epochs)


    def extract_classes(self):

        y_real = []

        for i in range(len(self.training_set)):

            if self.training_set[i][-1] == "spam":

                y_real.append(1)

            else:

                y_real.append(0)

            self.training_set[i].pop()

        return y_real


    def create_BOW(self):

        vocabulary_size = len(self.vocabulary)
        training_set_cardinality = len(self.training_set)
        
        shape = ((vocabulary_size, training_set_cardinality))

        X = pd.DataFrame(np.zeros(shape), columns=[f"Message {i}" for i in range(training_set_cardinality)], index=self.vocabulary)

        for word in self.vocabulary:

            for i in range(len(self.training_set)):

                if word in self.training_set[i]:

                    X.at[word, f"Message {i}"] += 1

        return X.to_numpy()


    def sigmoid(self, z):

        return 1 / (1 + np.exp(-z))

    
    def initialize_parameters(self, n_features):
        # Initialize the coefficients (weights) to 0
        self.weights = np.zeros(n_features)
        self.bias = 0
        return self.weights, self.bias
    

    def predict(self):
        
        linear_model = np.dot(self.X, self.weights) + self.bias

        y_predicted = self.sigmoid(linear_model)

        return y_predicted
    

    def compute_loss_and_gradient(self):

        m = self.X.shape[0]  # Number of messages
        y_predicted = self.predict()
        
        # Compute log-loss
        loss = - (1 / m) * np.sum(self.y_real * np.log(y_predicted) + (1 - self.y_real) * np.log(1 - y_predicted))
        
        # Gradient of weight and bias
        dw = (1 / m) * np.dot(X.T, (y_predicted - self.y_real))
        db = (1 / m) * np.sum(y_predicted - self.y_real)
        
        return loss, dw, db
    

    def gradient_descent(self, learning_rate, epochs):
        
        for i in range(epochs):
            #Compute loss and gradient
            loss, dw, db = self.compute_loss_and_gradient()
            
            # Update parameters
            self.weights -= learning_rate * dw
            self.bias -= learning_rate * db
        
        return self.weights, self.bias
    





