import os
import math
import numpy as np
import nltk
from nltk import word_tokenize
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from collections import Counter
default_imdb_folder_location = "../aclImdb"

def imdb_sentiment_reader(dataset_type='train', sentiment='both', folder_location=default_imdb_folder_location):
    """
    Iterator over the imdb dataset.
    Args:
        is_train: ['train', 'val', 'test] - whether to iterate over the train, val, or test sets.
        sentiment: ['pos', 'neg', 'both']: whether to iterate over just the positive, just the
                   negative, or both.
    Returns: Iterator over (filename, movie_review, sentiment_score) tuples. 
    """
    subfolder = 'train' if dataset_type=='train' else 'test'
    dataset_path = os.path.join(folder_location, subfolder)
    if sentiment=='pos' or sentiment=='both':
        # Sort by the index
        filenames = sorted(os.listdir(os.path.join(dataset_path, 'pos')), 
                                key=lambda filename: int(filename.split('_')[0]))
        # Take a slice if these are for val/test
        if dataset_type == 'val' or dataset_type == 'test':
            cutoff = int(math.ceil(len(filenames) * .2))
            if dataset_type == 'val':
                filenames = filenames[:cutoff]
            else:
                filenames = filenames[cutoff:]
        for filename in filenames:
            sentiment_score = int(filename.split('_')[1].split('.')[0])
            with open(os.path.join(dataset_path, 'pos', filename)) as f:
                review = f.read()
            yield filename, review.decode('utf-8'), sentiment_score
    if sentiment=='neg' or sentiment=='both':
        # Sort by the index 
        filenames = sorted(os.listdir(os.path.join(dataset_path, 'neg')), 
                                key=lambda filename: int(filename.split('_')[0]))
         # Take a slice if these are for val/test
        if dataset_type == 'val' or dataset_type == 'test':
            cutoff = int(math.ceil(len(filenames) * .2))
            if dataset_type == 'val':
                filenames = filenames[:cutoff]
            else:
                filenames = filenames[cutoff:]
        for filename in filenames:
            sentiment_score = int(filename.split('_')[1].split('.')[0])
            with open(os.path.join(dataset_path, 'neg', filename)) as f:
                review = f.read()
            yield filename, review.decode('utf-8'), sentiment_score

class DefaultEvaluator():
    """
    Default evaluator for the sentiment IMDB problem.
    Uses the default evaluation metric defined in the paper.
    """
    def __init__(self, verbose=False):
        self.verbose=verbose
        self.sentiment_analyzer = SentimentIntensityAnalyzer()
        self.sentence_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
        self.build_bigram_lists()
    
    def get_bigrams(self, tokens):
        # Gets bigrams from lowercased tokens
        return [(tokens[i].lower(), tokens[i+1].lower()) for i in range(len(tokens) - 1)]
    
    def build_bigram_lists(self):
        """
        Build lists of all positive and negative training bigrams.
        """
        self.pos_bigrams = set()
        print "Building positive bigram list..."
        for index, (filename, review, score) in enumerate(
            imdb_sentiment_reader(dataset_type='train', sentiment='pos')):
            try:
                tokens = word_tokenize(review)
                self.pos_bigrams.update(self.get_bigrams(tokens))
            except:
                print "Failed to tokenize: pos " + filename
            if self.verbose and index % 1000 == 0:
                print "Now on: " + str(index)
    
        print "Building negative bigram list..."     
        self.neg_bigrams = set()
        for index, (filename, review, score) in enumerate(
            imdb_sentiment_reader(dataset_type='train', sentiment='neg')):
            try:
                tokens = word_tokenize(review)
                self.neg_bigrams.update(self.get_bigrams(tokens))
            except:
                print "Failed to tokenize: neg " + filename
            if self.verbose and index % 1000 == 0:
                print "Now on: " + str(index)
                
    def _convert_imdb_sent_score(self, imdb_score):
        """
        Converts the IMDB sentiment score (on a 0-10 scale) to a binary (-1, 1) sentiment score.
        """
        # IMDB metric: positive is >=7, negative is <= 4
        if imdb_score >= 7:
            return 1.0
        elif imdb_score <= 4:
            return -1.0
        else:
            raise Exception('IMDB score not >=7 or <= 4')
    
    def _convert_vader_sent_score(self, vader_score):
        """
        Converts the vader score, a normalized score between -1 and 1 (where -1 is negative) 
        to a uniform mapping onto [1, 10]
        to its valence (negative scores are negative, positive are positive).
        """
        return np.interp(vader_score, [-1,1], [1,10])
    
    def _get_sentiment_changed_score(self, old_score, new_review):
        """
        Return a value between 0 and 1 denoting the degree of change in sentiment.
        """
        new_sent_score = self._convert_vader_sent_score(
            self.sentiment_analyzer.polarity_scores(new_review)['compound'])
        return (abs(new_sent_score - old_score) / max(old_score - 1, 10 - old_score))
    
    def _get_sentence_score(self, old_review, new_review):
        """
        Return a closeness score based on the similarity of number of sentences.
        """
        num_old_sentences = len(self.sentence_tokenizer.tokenize(old_review))
        num_new_sentences = len(self.sentence_tokenizer.tokenize(new_review))
        return min(num_old_sentences/float(num_new_sentences), num_new_sentences/float(num_old_sentences))
    
    def _get_proper_noun_score(self, tagged_old_review, tagged_new_review):
        """
        Returns a metric between 0-1 based on the overlap between proper nouns in the two reviews.
        """
        proper_nouns_old = set([token for token in tagged_old_review if token[1] == 'NNP'])
        proper_nouns_new = set([token for token in tagged_new_review if token[1] == 'NNP'])
        num_overlap = len(proper_nouns_old.intersection(proper_nouns_new))
        num_nouns = max(len(proper_nouns_old), len(proper_nouns_new))
        if num_nouns == 0:
            return 0
        else:
            return float(num_overlap) / max(len(proper_nouns_old), len(proper_nouns_new))
    
    def _get_pos_score(self, tagged_old_review, tagged_new_review):
        """
        Returns a value between 0 and 1 based on the multiset overlap between the POS tags in 
        both reviews.
        """
        pos_tags_old=[tagged_token[1] for tagged_token in tagged_old_review]
        pos_tags_new=[tagged_token[1] for tagged_token in tagged_new_review]
        intersection = Counter(pos_tags_old) & Counter(pos_tags_new)
        num_overlap = sum(intersection.values())
        num_pos = max(len(pos_tags_old), len(pos_tags_new))
        if num_pos == 0:
            return 0
        else:
            return float(num_overlap) / num_pos
    
    def _get_closeness_score(self, old_review, new_review, tokenized_old_review, tokenized_new_review):
        """
        Return a value between 0 and 1 evaluating how similar the old_review is in lingustic 
        structure to the new one.
        Calculated as (proper noun overlap) * (# sentences score) * (# POS score)
        """
        tagged_old_review = nltk.pos_tag(tokenized_old_review)
        tagged_new_review = nltk.pos_tag(tokenized_new_review)
        
        sentence_score = self._get_sentence_score(old_review, new_review)
        proper_noun_score = self._get_proper_noun_score(tagged_old_review, tagged_new_review)
        pos_score = self._get_pos_score(tagged_old_review, tagged_new_review)
        return sentence_score * proper_noun_score * pos_score
    
    def _get_typical_language_score(self, tokenized_old_review, old_score, tokenized_new_review):
        """
        Returns a value between 0 and 1 evaluating how similar the new_review is to existing reviews of 
        its desired sentiment.
        Concretely, evaluates the fraction of bigrams in the new_review that appear in either 
        all other training data reviews of the same sentiment, or within the old_review.
        """
        # Note: make sure to compare to other reviews that are the opposite of the old_score,
        # NOT the vader score.
        old_review_bigrams = set(self.get_bigrams(tokenized_old_review))
        new_review_bigrams = self.get_bigrams(tokenized_new_review)
        # Target sentiment is opposite of the old review sentiment
        target_sentiment = -1 * self._convert_imdb_sent_score(old_score) 
        num_similar_bigrams = 0
        for bigram in new_review_bigrams:
            if (bigram in old_review_bigrams) or \
            (target_sentiment > 0 and bigram in self.pos_bigrams) or \
            (target_sentiment < 0 and bigram in self.neg_bigrams):
                num_similar_bigrams += 1
        if len(new_review_bigrams) == 0:
            return 0
        else:
            return num_similar_bigrams / float(len(new_review_bigrams))
        
    def evaluate(self, filename, old_review, new_review, old_score):
        """
        Evaluation based on a sentiment changed score, closeness score, and 'typical language' score.
        """
        tokenized_old = word_tokenize(old_review)
        tokenized_new = word_tokenize(new_review)
        sent_change = self._get_sentiment_changed_score(old_score, new_review)
        closeness = self._get_closeness_score(old_review, new_review, tokenized_old, tokenized_new)
        typicality = self._get_typical_language_score(tokenized_old, old_score, tokenized_new)
        return sent_change * closeness * typicality


class ExperimentRunner():
    """
    Runs a sentiment experiment runner experiment. 
    Trains on the training set, then iterates over the reviews in the test set,
    transforming them using the transform_func and evaluating them using the eval_func.
    
    Outputs the average performance on the test set.
    
    Args:
        train_reader: an iterator over (filename, review, score) tuples.
        test_reader: an iterator over (filename, review, score) tuples.
        transform_func: should take (filename, review, score) and return a transformed string review.
        eval_func: should take (filename, old_review, new_review, old_sentiment_score) and return a score.
        verbose: default = False
    """
    def __init__(self, train_reader, test_reader, transform_func, evaluator=None, verbose=False):
        self.train_reader = train_reader
        self.test_reader = test_reader
        self.transform_func = transform_func
        if evaluator is None:
            self.evaluator = DefaultEvaluator()
            self.eval_func = self.evaluator.evaluate
        else:
            self.eval_func = evaluator.evaluate
        self.verbose = verbose
        self.scores = []
    
    def run_experiment(self):
        # Iterate over the test set, transforming the reviews and evaluating them
        for index, (filename, review, sent_score) in enumerate(self.test_reader):
            if self.verbose and index % 500 == 0:
                print "Now evaluating: " + str(index)
                # Print the running mean
                print "Current mean score: " + str(np.mean(self.scores))
            # Transform the review
            transformed_review = self.transform_func(filename, review, sent_score)
            # Evaluate the transformed review
            new_score = self.eval_func(filename, review, transformed_review, sent_score)
            self.scores.append(new_score)
        if self.verbose:
            print "Finished evaluating " + str(index) + " test reviews."
        print "Mean score: " + str(np.mean(self.scores))
        print "Median score: " + str(np.median(self.scores))
        print "Median score: " + str(np.std(self.scores))