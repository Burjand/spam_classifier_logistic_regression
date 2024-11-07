import random

class Validation_method():

    def __init__(self, dataset):

        self.dataset = dataset

    
    def hold_out(self, percentage_training_subset):

        training_set = []
        test_set = []

        messages_by_class = {
            "spam":[],
            "ham":[]
        }

        for message in self.dataset:

            if message[-1] == "spam":

                messages_by_class["spam"].append(message)

            else:

                messages_by_class["ham"].append(message)
        
        
        # Shuffle patterns in classes
        messages_by_class_shuffled = {}

        for label in messages_by_class.keys():

            patterns_in_class = len(messages_by_class[label])
            indexes_of_patterns = list(range(patterns_in_class))

            random.shuffle(indexes_of_patterns)

            for index in indexes_of_patterns:

                if label not in list(messages_by_class_shuffled.keys()):

                    messages_by_class_shuffled[label] = [messages_by_class[label][index]]

                else:
                    
                    messages_by_class_shuffled[label].append(messages_by_class[label][index]) 


        # Create training and test subsets
        for label in list(messages_by_class_shuffled.keys()):

            amount_of_patterns = len(messages_by_class_shuffled[label])
            cardinality_training = round((percentage_training_subset/100) * amount_of_patterns)

            for i in range(len(messages_by_class_shuffled[label])):

                if (i < cardinality_training):
                    
                    training_set.append(list(messages_by_class_shuffled[label][i]))

                else:

                    test_set.append(list(messages_by_class_shuffled[label][i]))

        return [training_set, test_set]