import os
import config
from Preprocessing import Preprocessing
from Validation_method import Validation_method
from Logistic_regression_classifier import Logistic_regression_classifier
from Classifier_tester import Classifier_tester

if __name__ == "__main__":

    # Paths
    raw_files_path = config.raw_files_path
    processed_files_path = config.processed_files_path

    corpus_file = config.corpus_file
    stopwords_file = config.stopwords_file
    encoding = config.encoding

    # Preprocessing
    preprocessed_text_object = Preprocessing(raw_files_path + corpus_file, raw_files_path + stopwords_file)

    preprocessed_messages = preprocessed_text_object.lemmatized_messages
    vocabulary = preprocessed_text_object.vocabulary

    # Division in training and test sets
    training_set, test_set = Validation_method(preprocessed_messages).hold_out(80)  #Applied Hold Out as validation method with 80% of the patterns assigned to the training set

    # CLASSIFIER
    # Train model
    classifier = Logistic_regression_classifier(vocabulary, training_set)
    
    # Test classifier (Obtain all the messages classified)
    classifier_tester = Classifier_tester(vocabulary, test_set, classifier)

    print(classifier_tester.y_predicted)
    



