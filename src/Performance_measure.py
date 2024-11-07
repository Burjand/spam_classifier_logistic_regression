"""
This class receives a classified test set (Each element includes the pattern + the actual class + the assigned class) and generates the confusion matrix
"""

import math

class Performance_measure():

    def __init__(self, y_real, y_predicted, positive_class=1):

        self.y_real = y_real
        self.y_predicted = y_predicted
        self.positive_class = positive_class

        self.confusion_matrix = self.generate_confusion_matrix()

        self.accuracy = self.calculate_accuracy()
        self.error_rate = self.calculate_error_rate()
        self.sensitivity = self.calculate_sensitivity()
        self.specificity = self.calculate_specificity()
        self.precision = self.calculate_precision()
        self.balanced_accuracy = self.calculate_balanced_accuracy()
        self.f1_score = self.calculate_f1_score()
        self.mcc = self.calculate_MCC()
        self.informedness = self.calculate_informedness()
        self.fallout = self.calculate_fallout()


    def generate_confusion_matrix(self):

        confusion_matrix = {'TP': 0, 'FN': 0, 'FP': 0, 'TN': 0}

        for i in range(len(self.y_predicted)):

            if (self.y_predicted[i] == self.positive_class and self.y_predicted[i] == self.y_real[i]):

                confusion_matrix['TP'] += 1

            elif (self.y_predicted[i] == self.positive_class and self.y_predicted[i] != self.y_real[i]):

                confusion_matrix['FP'] += 1

            elif (self.y_predicted[i] != self.positive_class and self.y_predicted[i] == self.y_real[i]):

                confusion_matrix['TN'] += 1

            elif (self.y_predicted[i] != self.positive_class and self.y_predicted[i] != self.y_real[i]):

                confusion_matrix['FN'] += 1

        return confusion_matrix
    

    def calculate_imbalance_ratio(self):

        label_count = {}

        for label in self.y_real:

            if label not in label_count.keys():

                label_count[label] = 1

            else:

                label_count[label] += 1

        smallest_class_cardinality = label_count[min(label_count, key=label_count.get)]
        biggest_class_cardinality = label_count[max(label_count, key=label_count.get)]

        return biggest_class_cardinality/smallest_class_cardinality


    def calculate_accuracy(self):

        IR = self.calculate_imbalance_ratio()

        if IR <= 1.5:

            TP = self.confusion_matrix['TP']
            FN = self.confusion_matrix['FN']
            FP = self.confusion_matrix['FP']
            TN = self.confusion_matrix['TN']

            return (TP + TN) / (TP + FN + FP + TN)
        
        else:

            return f"Accuracy is not an appropiate measure because test set is unbalanced (IR = {IR})"
        
    
    def calculate_error_rate(self):

        IR = self.calculate_imbalance_ratio()

        if IR <= 1.5:

            TP = self.confusion_matrix['TP']
            FN = self.confusion_matrix['FN']
            FP = self.confusion_matrix['FP']
            TN = self.confusion_matrix['TN']

            return (FN + FP) / (TP + FN + FP + TN)
        
        else:

            return f"Error rate is not an appropiate measure because test set is unbalanced (IR = {IR})"


    def calculate_sensitivity(self):

        TP = self.confusion_matrix['TP']
        FN = self.confusion_matrix['FN']

        return TP / (TP + FN)
    

    def calculate_specificity(self):
        
        FP = self.confusion_matrix['FP']
        TN = self.confusion_matrix['TN']

        return TN / (FP + TN)


    def calculate_precision(self):

        TP = self.confusion_matrix['TP']
        FP = self.confusion_matrix['FP']

        try:

            return TP / (TP + FP)
        
        except(ZeroDivisionError):

            return f"Precision is not an appropiate measure because it leads to division by zero"


    def calculate_balanced_accuracy(self):

        sensitivity = self.sensitivity
        specificity = self.specificity

        return (sensitivity + specificity) / 2


    def calculate_f1_score(self):

        precision = self.precision
        sensitivity = self.sensitivity

        if type(precision) is float:

            return 2 / ((1 / sensitivity) + (1 / precision))
        
        else:

            return f"f1_score is not an appropiate measure because it leads to division by zero"


    def calculate_MCC(self):

        try:

            TP = self.confusion_matrix['TP']
            FN = self.confusion_matrix['FN']
            FP = self.confusion_matrix['FP']
            TN = self.confusion_matrix['TN']

            return ((TN*TP - FN*FP) / math.sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN)))
        
        except(ZeroDivisionError):

            return f"Matthews Correlation Coefficient is not an appropiate measure because it leads to division by zero"


    def calculate_informedness(self):

        return (self.sensitivity + self.specificity) - 1


    def calculate_fallout(self):

        FP = self.confusion_matrix['FP']
        TN = self.confusion_matrix['TN']

        return FP / (FP + TN)


