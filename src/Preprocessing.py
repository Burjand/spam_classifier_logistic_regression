import nltk
import spacy
from wordsegment import load, segment
import re

class Preprocessing():

    def __init__(self, corpus_file, stopwords_file=None, encoding="UTF-8"):

        # Extract raw text
        self.raw_messages = self.extract_text_from_file(corpus_file, encoding)

        #Lower case
        self.lower_cased_messages = self.lower_case(self.raw_messages)
        
        # Tokenize
        self.tokenized_messages = self.tokenize(self.lower_cased_messages)

        # Clean text from punctuation marks
        self.messages_no_punctuation = self.remove_punctuation_marks_and_blanks(self.tokenized_messages)

        #Deal with joined words that are not supposed to be joined
        self.messages_no_punctuation_no_joined_words = self.split_joined_words(self.messages_no_punctuation)

        # Clean text from stopwords
        self.messages_no_stopwords_no_punctuation = self.remove_stopwords(self.messages_no_punctuation_no_joined_words, stopwords_file, encoding)
        
        # Create spacy document and lemmatize
        self.lemmatized_messages = self.lemmatize_messages(self.messages_no_stopwords_no_punctuation)

        # Vocabulary
        self.vocabulary = self.obtain_vocabulary(self.lemmatized_messages)

    


    def extract_text_from_file(self, file, encoding):

        file = open(file, "r", encoding=encoding, errors="ignore")
        raw_text = file.readlines()
        file.close()

        return raw_text

    
      
    def lower_case(self, text_to_lower):

        lower_cased_messages = []

        for element in text_to_lower:

            lower_cased_messages.append(element.lower())

        return lower_cased_messages


    
    def tokenize(self, messages_to_tokenize):

        tokenized_messages = []

        for message in messages_to_tokenize:

            tokenized_messages.append(nltk.word_tokenize(message))

        return(tokenized_messages)



    def remove_punctuation_marks_and_blanks(self, raw_messages):

        messages_no_punctuation = []

        for sms in raw_messages:

            temp_message = []

            for item in sms:

                cleaned_item = ""
                for character in item:

                    if (character.isalpha()):

                        cleaned_item += character

                if cleaned_item != "":

                    temp_message.append(cleaned_item)

            messages_no_punctuation.append(temp_message)

        return messages_no_punctuation



    def split_joined_words(self, messages_no_punctuation):

        load()

        cleaned_messages = []

        for message in messages_no_punctuation:

            temp_message = []

            for item in message:

                temp_message.append(" ".join(segment(item)))

            cleaned_messages.append(temp_message)

        return cleaned_messages



    def remove_stopwords(self, messages_no_punctuation, stopwords_file, encoding):

        file = open(stopwords_file, "r", encoding=encoding)
        stopwords_raw = file.readlines()
        file.close()

        stopwords_list = [word.replace("\n","") for word in stopwords_raw]

        sentences_no_stopwords = []

        for sentence in messages_no_punctuation:

            temp_sentence = []

            for item in sentence:

                if item not in stopwords_list:

                    temp_sentence.append(item)
            
            sentences_no_stopwords.append(temp_sentence)

        return sentences_no_stopwords   


    
    def lemmatize_messages(self, sentences_no_stopwords_no_punctuation):

        lemmatized_messages = []

        nlp = spacy.load('en_core_web_sm')
        
        for sentence in sentences_no_stopwords_no_punctuation:

            text_to_lemmatize = " ".join([word for word in sentence])            

            doc = nlp(text_to_lemmatize)

            lemmatized_messages.append([token.lemma_ for token in doc])

        return lemmatized_messages




    def obtain_vocabulary(self, lemmatized_messages):

        # Obtain vocabulary
        lemmatized_text = []

        for message in lemmatized_messages:

            for token in message[:-1]:

                lemmatized_text.append(token)
        

        vocabulary = sorted(list(set(lemmatized_text)))
                

        return vocabulary
    