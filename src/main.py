import os
import config
from Preprocessing import Preprocessing
from Validation_method import Validation_method
from Logistic_regression_classifier import Logistic_regression_classifier
from Classifier_tester import Classifier_applier
from Performance_measure import Performance_measure

if __name__ == "__main__":

    # Paths
    raw_files_path = config.raw_files_path
    processed_files_path = config.processed_files_path

    corpus_file = config.corpus_file
    stopwords_file = config.stopwords_file
    encoding = config.encoding

    # Dataset preprocessing
    preprocessed_text_object = Preprocessing(raw_files_path + corpus_file, raw_files_path + stopwords_file, encoding='latin-1')

    preprocessed_messages = preprocessed_text_object.lemmatized_messages
    vocabulary = preprocessed_text_object.vocabulary

    # Validation method
    training_set, test_set = Validation_method(preprocessed_messages).hold_out(80)  #Applied Hold Out as validation method with 80% of the patterns assigned to the training set

    # ALGORITHM
    # Train model
    classifier = Logistic_regression_classifier(vocabulary, training_set)
    
    # Classify test set (Obtain all the messages classified)
    classifier_applier = Classifier_applier(vocabulary, test_set, classifier)

    test_set_y_real = classifier_applier.y_real
    test_set_y_predicted = classifier_applier.y_predicted

    performance_measures = Performance_measure(test_set_y_real, test_set_y_predicted)

    print(f"Confusion matrix: {performance_measures.confusion_matrix}")
    print(f"Accuracy: {performance_measures.accuracy}")
    print(f"Error rate: {performance_measures.error_rate}")
    print(f"Sensitivity: {performance_measures.sensitivity}")
    print(f"Specificity: {performance_measures.specificity}")
    print(f"Precision: {performance_measures.precision}")
    print(f"Balanced Accuracy: {performance_measures.balanced_accuracy}")
    print(f"F1 score: {performance_measures.f1_score}")
    print(f"MCC: {performance_measures.mcc}")
    print(f"Informedness: {performance_measures.informedness}")
    print(f"Fallout: {performance_measures.fallout}")

