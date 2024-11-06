import pandas as pd
import numpy as np

class Classifier_tester():

    def __init__(self, vocabulary, test_set, classifier):

        self.vocabulary = vocabulary
        self.test_set = test_set
        self.classifier = classifier

        self.y_real = self.extract_classes()

        self.X_test_set = self.create_BOW()

        self.y_predicted = self.classify_test_set()


    def extract_classes(self):

        y_real = []

        for i in range(len(self.test_set)):

            if self.test_set[i][-1] == "spam":

                y_real.append(1)

            else:

                y_real.append(0)

            self.test_set[i].pop()

        return y_real
    

    def create_BOW(self):

        vocabulary_size = len(self.vocabulary)
        test_set_cardinality = len(self.test_set)
        
        shape = ((vocabulary_size, test_set_cardinality))

        X_test_set = pd.DataFrame(np.zeros(shape), columns=[f"Message {i}" for i in range(test_set_cardinality)], index=self.vocabulary)

        for word in self.vocabulary:

            for i in range(len(self.test_set)):

                if word in self.test_set[i]:

                    X_test_set.at[word, f"Message {i}"] += 1

        return X_test_set.to_numpy()
    

    def classify_test_set(self):

        y_predicted = []

        for message in self.X_test_set:

            label = self.classify_message(message)

            y_predicted.append(label)

        return y_predicted


    def classify_message(self, message_features):

        probability = self.classifier.predict(message_features, self.classifier.weights, self.classifier.bias)

        return 1 if probability >= 0.5 else 0  # 1 spam, 0 ham