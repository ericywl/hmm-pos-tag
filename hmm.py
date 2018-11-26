#!/usr/bin/env python3
"""ML Design Project"""

from collections import deque
from operator import itemgetter
import os


class HMM:
    """HMM class"""
    UNKNOWN_TOKEN = "UNK"

    def __init__(self, states):
        """
        Given states should have START and STOP (or equivalent)
        as the first two element

        Variables:
        end_states -- array of START and STOP states (or equivalent)
        states -- array that stores the rest of the states
        emission_probs -- dictionary of state:emission_probabilities
        transition_probs -- dictionary of state:transition_probabilities
        training_words -- set of words that have occured in training set
        """
        self.end_states = states[0:2]
        self.states = states[2:]
        self.emission_probs = {}
        self.transition_probs = {}
        self.training_words = set()

    def train(self, filename):
        """
        Train the model using supervised learning on the given
        training set file

        Arguments:
        filename -- name of the training set file
        """
        observations = self.process_file(filename, data_type="train")
        self.emission_probs = self.estimate_emission(observations)
        self.transition_probs = self.estimate_transition(observations)

    # TODO: Add in Viterbi, currently only predict naively
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
            predictions.append(self.naive_label_sequence(sequence))
        with open(out_filename, "w") as out_file:
            for pred_deque in predictions:
                while pred_deque:
                    word, state = pred_deque.popleft()
                    if state in self.end_states:
                        continue
                    out_file.write(f"{word} {state}\n")
                out_file.write("\n")

    @staticmethod
    def _calculate_emission_mle(word_count, state_count, smooth_k):
        """
        Use MLE and smoothing to calculate emission probability

        Arguments:
        word_count -- number of emissions of word from state
        state_count -- number of occurences of state in observations
        smooth_k -- smoothing variable

        Returns:
        emission_mle -- emission probability of word from state
        """
        return float(word_count) / (state_count + smooth_k)

    def estimate_emission(self, observations, smooth_k=1):
        """
        Estimate the emission probabilities given observations

        Arguments:
        observations -- training data, an array of deque with 
                        word-state tuples
        smooth_k -- smoothing variable

        Returns:
        emission_probs -- dictionary of emission probabilities
        """
        state_emission_counts = {}
        for state in self.states:
            state_emission_counts[state] = {}
        for ws_deque in observations:
            for word, state in ws_deque:
                self._check_end_states(ws_deque)
                if state in self.end_states:
                    continue
                if state not in self.states:
                    raise Exception(f"Invalid state in data: {state}")
                temp_word = word.lower()
                # Add word to training_words for use in smoothing
                self.training_words.add(temp_word)
                if word not in state_emission_counts[state]:
                    state_emission_counts[state][temp_word] = 0
                state_emission_counts[state][temp_word] += 1
        emission_probs = {}
        for state, emission_counts in state_emission_counts.items():
            # Sum all emission counts of a particular state
            state_cnt = sum(emission_counts.values())
            if state not in emission_probs:
                emission_probs[state] = {}
            # Calculate emission MLE for each state
            emission_probs[state] = {
                word: HMM._calculate_emission_mle(
                    word_cnt, state_cnt, smooth_k
                ) for word, word_cnt in emission_counts.items()
            }
            # Introduce UNK word token with smoothing
            emission_probs[state][HMM.UNKNOWN_TOKEN] \
                = HMM._calculate_emission_mle(smooth_k, state_cnt, smooth_k)
        return emission_probs

    @staticmethod
    def _calculate_transition_mle(next_state_cnt, state_cnt):
        """
        Use MLE to calculate the transition probability

        Arguments:
        next_state_cnt -- number of transitions from state to next_state
        state_cnt -- number of occurences of state in observations

        Returns:
        transition_mle -- transition probability of state to next_state
        """
        return float(next_state_cnt) / state_cnt

    def estimate_transition(self, observations):
        """
        Estimate the transition probabilities given observations

        Arguments:
        observations -- training data, an array of deque with 
                        word-state tuples

        Returns:
        transition_probs -- dictionary of transition probabilities
        """
        state_transition_counts = {self.end_states[0]: {}}
        for state in self.states:
            state_transition_counts[state] = {}
        for ws_deque in observations:
            self._check_end_states(ws_deque)
            for i in range(len(ws_deque)):
                if i == len(ws_deque) - 1:
                    continue
                curr_state = ws_deque[i][1]
                next_state = ws_deque[i + 1][1]
                if next_state not in state_transition_counts[curr_state]:
                    state_transition_counts[curr_state][next_state] = 0
                state_transition_counts[curr_state][next_state] += 1
        transition_probs = {}
        for curr_state, transition_counts in state_transition_counts.items():
            # Sum all transition counts of a particular state
            state_cnt = sum(transition_counts.values())
            if curr_state not in transition_probs:
                transition_probs[curr_state] = {}
            # Calculate transition MLE for each state
            transition_probs[curr_state] = {
                next_state: HMM._calculate_transition_mle(
                    next_state_cnt, state_cnt
                ) for next_state, next_state_cnt in transition_counts.items()
            }
        return transition_probs

    def _argmax_emission(self, word):
        """
        Get the most likely state given a word

        Arguments:
        word -- word from a sequence

        Returns:
        state -- state with highest probability for word in
                 emission_probs
        """
        if not self.emission_probs:
            raise Exception("Emission probabilities is empty")
        word_emission_probs = []
        for state, word_probs in self.emission_probs.items():
            temp_word = word.lower()
            if temp_word not in self.training_words:
                curr_prob = word_probs[HMM.UNKNOWN_TOKEN]
            else:
                curr_prob = word_probs.get(temp_word, 0)
            word_emission_probs.append((state, curr_prob))
        return max(word_emission_probs, key=itemgetter(1))[0]

    def naive_label_sequence(self, sequence):
        """
        Attach predicted label to the given sequence, with Naive-Bayes
        assumption ie. independence of words

        Arguments:
        sequence -- deque with word-state tuples, but with empty states 
                    except START and STOP

        Returns:
        labelled_sequence -- new deque with all word-states filled
        """
        if not self.emission_probs:
            raise Exception("Emission probabilities is empty")
        prediction = deque([("", self.end_states[0])])
        for word, state in sequence:
            if state in self.end_states:
                continue
            prediction.append((word, self._argmax_emission(word)))
        prediction.append(("", self.end_states[1]))
        return prediction

    def _dp_helper(self, iteration, state, sequence, viterbi_graph):
        """
        Dynamic programming helper function for Viterbi algorithm,
        modifies viterbi_graph
        
        Arguments:
        iteration -- current iteration
        state -- current state
        sequence -- deque with word-state tuples, but with empty states 
                    except START and STOP
        viterbi_graph -- dictionary representing incomplete Viterbi graph
        """
        # scaling constant to avoid arithmetic underflow if it occurs
        scaling_constant = 1e3
        if iteration == 0:
            raise Exception(f"Invalid iteration: {iteration}")
        if iteration != len(sequence) - 1 and state == self.end_states[1]:
            # STOP encountered but we have not reached the end yet
            # (This is intended)
            return
        word = sequence[iteration][0].lower()
        word = word if word in self.training_words else HMM.UNKNOWN_TOKEN
        if iteration == 1:
            # Base case
            alpha = \
                self.transition_probs[self.end_states[0]].get(state, 0)
            beta = self.emission_probs[state].get(word, 0)
            viterbi_graph[iteration][state] \
                = (self.end_states[0], alpha * beta * scaling_constant)
            return
        # Forward recursion
        temp_nodes = []
        for prev_state in self.states:
            prev_optimal_prob = viterbi_graph[iteration - 1][prev_state][1]
            alpha = self.transition_probs[prev_state].get(state, 0)
            if iteration == len(sequence) - 1:
                beta = 1
            else:
                beta = self.emission_probs[state].get(word, 0)
            prod = prev_optimal_prob * alpha * beta * scaling_constant
            temp_nodes.append((prev_state, prod))
        viterbi_graph[iteration][state] = max(temp_nodes, key=itemgetter(1))

    def viterbi(self, sequence):
        """
        Construct Viterbi graph using sequence and model parameters
        
        Arguments:
        sequence -- deque with word-state tuples, but with empty state
                    except START and STOP
        
        Returns:
        viterbi_graph -- dictionary representing the resulting Viterbi graph
        """
        self._check_end_states(sequence)
        viterbi_graph = {i: {} for i in range(1, len(sequence))}
        for iteration in range(len(sequence)):
            if iteration == 0:
                continue
            for state in self.states + [self.end_states[1]]:
                self._dp_helper(iteration, state, sequence, viterbi_graph)
        return viterbi_graph

    def process_file(self, filename, data_type):
        """
        Process the file into an array of deques with word-state pair

        Arguments:
        filename -- name of the file
        data_type -- specify training or testing set

        Returns:
        data -- array of deques with word-state tuples;
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
                    if data_type == "test":
                        if len(split_ws) > 1:
                            raise Exception("Wrong testing set format")
                        split_ws.append("")
                    word_state_deque.append(tuple(split_ws))
                word_state_deque.appendleft(("", self.end_states[0]))
                word_state_deque.append(("", self.end_states[1]))
                data.append(word_state_deque)
            return data

    def _check_end_states(self, ws_deque):
        """Check that end two ends of word-state deque have correct states"""
        state = ws_deque[0][1]
        if state != self.end_states[0]:
            raise Exception(f"Invalid starting state: {state}")
        state = ws_deque[len(ws_deque) - 1][1]
        if state != self.end_states[1]:
            raise Exception(f"Invalid stopping state: {state}")
        return True


if __name__ == "__main__":
    EN_STATES = [
        "START", "STOP",
        "B-VP", "I-VP",
        "B-NP", "I-NP",
        "B-PP", "I-PP",
        "B-INTJ", "I-INTJ",
        "B-ADJP", "I-ADJP",
        "B-SBAR", "I-SBAR",
        "B-ADVP", "I-ADVP",
        "B-CONJP", "I-CONJP",
        "O", "B-PRT"
    ]
    OTHER_STATES = [
        "START", "STOP",
        "B-positive", "I-positive",
        "B-neutral", "I-neutral",
        "B-negative", "I-negative",
        "O"
    ]
    hmm = HMM(EN_STATES)
    hmm.train("EN/train")
    # hmm.predict("EN/dev.in")
    data = hmm.process_file("EN/dev.in", data_type="test")
    sample = data[0]
    print(hmm.viterbi(sample))
