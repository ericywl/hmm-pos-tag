#!/usr/bin/env python3
"""ML Design Project"""

from collections import deque

class HMM:
    """HMM class"""

    def __init__(self, states):
        self.states = states
        self.emission_probabilities = {}
        self.transition_probabilities = {}
        
    def train(self, filename):
        """
        Train the model using supervised learning on the given
        training set file
        
        Arguments:
        filename -- name of the training set file
        """
        observations = HMM.process_file(filename, data_type="train")
        self.emission_probabilities = self.estimate_emission(observations)
        
    @staticmethod
    def _calculate_emission_mle(num_emissions, num_state, smoothing_k=1):
        return float(num_emissions) / (num_state + smoothing_k)

    def estimate_emission(self, observations):
        """
        Estimate the emission probabilities given observations
        
        Arguments:
        observations -- training data, an array of deque with 
                        word-state pair tuple

        Returns:
        emi_probs -- a dictionary of emission probabilities
        """
        sw_counts = {}
        for state in self.states:
            if state in ["START", "STOP"]:
                continue
            sw_counts[state] = {}
        for ws_deque in observations:
            for word, state in ws_deque:
                if state not in self.states:
                    raise Exception("Invalid state in data")
                if state in ["START", "STOP"]:
                    continue
                if word not in sw_counts[state]:
                    sw_counts[state][word] = 0
                sw_counts[state][word] += 1
        emi_probs = {}
        for state, word_counts in sw_counts.items():
            num_state = sum(word_counts.values())
            if state not in emi_probs:
                emi_probs[state] = {}
            emi_probs[state] = {
                word: HMM._calculate_emission_mle(count, num_state) \
                for word, count in word_counts.items()
            }
        return emi_probs
        
    def label_sequence(self, sequence):
        if not self.emission_probabilities:
            raise Exception("Emission probabilities is empty")

    @staticmethod
    def process_file(filename, data_type):
        """
        Process the file into an array of deques with word-state pair

        Arguments:
        filename -- name of the file
        data_type -- specify training or testing set

        Returns:
        data -- array of deques with word-state pair tuple
        """
        if data_type.lower() not in ["train", "test"]:
            raise Exception("Invalid data type given")
        with open(filename) as file:
            sentences = file.read().rstrip().split("\n\n")
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
                word_state_deque.appendleft(("", "START"))
                word_state_deque.append(("", "STOP"))
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
    print(hmm.emission_probabilities)
