#!/usr/bin/env python3
"""ML Design Project"""

from collections import deque
from operator import itemgetter
import os

class HMM:
    """HMM class"""
    UNKNOWN_TOKEN = "UNK"

    def __init__(self, states, emi_probs={}, trans_probs={}):
        """Given states should have START and STOP (or equivalent)
        as the first two element"""
        self.end_states = states[0:2]
        self.states = states[2:]
        self.emission_probabilities = emi_probs
        self.transition_probabilities = trans_probs
        self.training_words = set()
        
    def train(self, filename):
        """
        Train the model using supervised learning on the given
        training set file
        
        Arguments:
        filename -- name of the training set file
        """
        observations = self.process_file(filename, data_type="train")
        self.emission_probabilities = self.estimate_emission(observations)
        
    def predict(self, in_filename, out_filename=None):
        """
        Predict the word tags using decoding on the given
        testing set file
        
        Arguments:
        in_filename -- name of the testing set file 
        out_filename -- name of predictions output file
        """
        dir_path = os.path.dirname(os.path.realpath(in_filename))
        if out_filename == None:
            out_filename = os.path.join(dir_path, "dev.test.out") 
        sequences = self.process_file(in_filename, data_type="test")
        predictions = []
        for sequence in sequences:
            predictions.append(self.label_sequence(sequence))
        with open(out_filename, "w") as out_file:
            for pred_deque in predictions:
                while pred_deque:
                    word, state = pred_deque.popleft()
                    if state in self.end_states:
                        continue
                    out_file.write(f"{word} {state}\n")
                out_file.write("\n")
        
    @staticmethod
    def _calculate_emission_mle(num_emissions, num_state, smooth_k):
        """
        Use MLE and smoothing to calculate emission probability
        
        Arguments:
        num_emissions -- number of emissions of word from state
        num_state -- number of occurences of state in observations
        smooth_k -- smoothing variable
        
        Returns:
        emi_prob -- emission probability of word from state
        """
        return float(num_emissions) / (num_state + smooth_k)

    def estimate_emission(self, observations, smooth_k=1):
        """
        Estimate the emission probabilities given observations
        
        Arguments:
        observations -- training data, an array of deque with 
                        word-state pair tuple
        smooth_k -- smoothing variable

        Returns:
        emi_probs -- a dictionary of emission probabilities
        """
        sw_counts = {}
        for state in self.states:
            sw_counts[state] = {}
        for ws_deque in observations:
            for word, state in ws_deque:
                if state in self.end_states:
                    continue
                if state not in self.states:
                    raise Exception("Invalid state in data")
                self.training_words.add(word)
                if word not in sw_counts[state]:
                    sw_counts[state][word] = 0
                sw_counts[state][word] += 1
        emi_probs = {}
        for state, word_counts in sw_counts.items():
            num_state = sum(word_counts.values())
            if state not in emi_probs:
                emi_probs[state] = {}
            emi_probs[state] = {
                word: HMM._calculate_emission_mle(cnt, num_state, smooth_k) \
                    for word, cnt in word_counts.items()
            }
            emi_probs[state][HMM.UNKNOWN_TOKEN] \
                = HMM._calculate_emission_mle(smooth_k, num_state, smooth_k)
        return emi_probs
        
    def _argmax_emission(self, word):
        """
        Get the most likely state given a word
        
        Arguments:
        word -- word from a sequence
        
        Returns:
        state -- state with highest probability for word in
                 emission_probabilities
        """
        if not self.emission_probabilities:
            raise Exception("Emission probabilities is empty")
        word_emission_probs = []
        for state, word_probs in self.emission_probabilities.items():
            if word not in self.training_words:
                curr_prob = word_probs[HMM.UNKNOWN_TOKEN]
            else:
                curr_prob = word_probs[word] if word in word_probs else 0
            word_emission_probs.append((state, curr_prob))
        print(word, word_emission_probs)
        return max(word_emission_probs, key=itemgetter(1))[0]
        
    def label_sequence(self, sequence):
        if not self.emission_probabilities:
            raise Exception("Emission probabilities is empty")
        prediction = deque([("", self.end_states[0])])
        for word, state in sequence:
            if state in self.end_states:
                continue
            prediction.append((word, self._argmax_emission(word)))
        prediction.append(("", self.end_states[1]))
        return prediction

    def process_file(self, filename, data_type):
        """
        Process the file into an array of deques with word-state pair

        Arguments:
        filename -- name of the file
        data_type -- specify training or testing set

        Returns:
        data -- array of deques with word-state pair tuple;
                for testing, the second element in the tuple is empty string
        """
        if data_type.lower() not in ["train", "test"]:
            raise Exception("Invalid data type given")
        with open(filename, "r") as data_file:
            sentences = data_file.read().rstrip().split("\n\n")
            data = []
            for sentence in sentences:
                word_state_deque = deque()
                for ws_pair in sentence.split("\n"):
                    split_ws = ws_pair.split(" ")
                    split_ws[0] = split_ws[0].lower()
                    if data_type == "test":
                        if len(split_ws) > 1:
                            raise Exception("Wrong testing set format")
                        split_ws.append("")
                    word_state_deque.append(tuple(split_ws))
                word_state_deque.appendleft(("", self.end_states[0]))
                word_state_deque.append(("", self.end_states[1]))
                data.append(word_state_deque)
            return data



if __name__ == "__main__":
    EN_STATES = [
        'START',
        'STOP',
        'B-VP',
        'I-VP',
        'B-NP',
        'I-NP',
        'B-PP',
        'I-PP',
        'O',
        'B-INTJ',
        'I-INTJ',
        'B-PRT',
        'B-ADJP',
        'I-ADJP',
        'B-SBAR',
        'I-SBAR',
        'B-ADVP',
        'I-ADVP',
        'B-CONJP',
        'I-CONJP'
    ]
    hmm = HMM(EN_STATES)
    hmm.train("EN/train")
    hmm.predict("EN/dev.in")
