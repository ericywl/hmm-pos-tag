#!/usr/bin/env python3
"""First order Hidden Markov Model"""

import copy
import os
import math
import subprocess
from collections import deque
from operator import itemgetter

from hmm import HMM


class HMM2(HMM):
    """Second order HMM class"""

    def __init__(self, states):
        super().__init__(states)
        self.transition_probs2 = {}

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
        # self.transition_probs2 = self.estimate_transition2(observations)

    def viterbi_predict(self, sequences):
        """
        Predict the word tags using decoding with Viterbi on the
        given sequences

        Arguments:
        sequences -- array of deque with word-state pairs, but the states
                     are empty except for START and STOP
        """
        predictions = []
        start_state = self.end_states[0]
        stop_state = self.end_states[1]
        for sequence in sequences:
            viterbi_graph = self.viterbi(sequence)
            labelled_sequence = deque([("", self.end_states[1])])
            temp_nodes = []
            for state in self.states:
                temp_tup = viterbi_graph[len(sequence) - 1][(state, stop_state)]
                temp_nodes.append(((temp_tup[0], state), temp_tup[1]))
            optimal_end_tup = max(temp_nodes, key=itemgetter(1))[0]
            labelled_sequence.appendleft((sequence[len(sequence) - 2][0],
                                          optimal_end_tup[1]))
            labelled_sequence.appendleft((sequence[len(sequence) - 3][0],
                                          optimal_end_tup[0]))
            for i in range(len(sequence) - 4, 0, -1):
                # Given
                word, _ = sequence[i]
                parent_node = viterbi_graph[i + 2][optimal_end_tup]
                optimal_end_tup = (parent_node[0], optimal_end_tup[0])
                labelled_sequence.appendleft((word, parent_node[0]))
            labelled_sequence.appendleft(("", start_state))
            predictions.append(labelled_sequence)
        return predictions

    #Deprecated
    def estimate_emission2(self, observations, smooth_k=1):
        """
        Estimate the emission probabilities given observations

        Arguments:
        observations -- training data, an array of deque with word-state
                        tuples
        smooth_k -- smoothing variable

        Returns:
        emission_probs -- dictionary of second-order emission probabilities
        """
        state_emission2_cnts = {}
        for state1 in [self.end_states[0]] + self.states:
            state_emission2_cnts[state1] = {
                state2: {} for state2 in self.states
            }
        for ws_deque in observations:
            self._check_end_states(ws_deque)
            for i, ws_pair in enumerate(ws_deque):
                if i == 0:
                    continue
                word, curr_state = ws_pair
                prev_state = ws_deque[i - 1][1]
                if curr_state in self.end_states:
                    continue
                for state in [prev_state, curr_state]:
                    if state not in [self.end_states[0]] + self.states:
                        raise Exception(f"Invalid state in data: {state}")
                # Add word to training_words for use in smoothing
                self.training_words.add(word)
                if word not in state_emission2_cnts[prev_state][curr_state]:
                    state_emission2_cnts[prev_state][curr_state][word] = 0
                state_emission2_cnts[prev_state][curr_state][word] += 1
        emission_probs2 = copy.deepcopy(state_emission2_cnts)
        for prev_state, curr_state_emissions in state_emission2_cnts.items():
            for curr_state, emission_counts in curr_state_emissions.items():
                # Sum all emission counts of a particular bigram
                state_cnt = sum(emission_counts.values())
                # Calculate emission MLE for each bigram
                emission_probs2[prev_state][curr_state] = {
                    word: self._calculate_emission_mle(
                        emission_cnt, state_cnt, smooth_k
                    ) for word, emission_cnt in emission_counts.items()
                }
                # Introduce UNK word token with smoothing
                emission_probs2[prev_state][curr_state][HMM.UNKNOWN_TOKEN] \
                    = self._calculate_emission_mle(smooth_k, state_cnt, smooth_k)
        return emission_probs2

    # Deprecated
    def get_emission2_probability(self, state1, state2, word):
        return (self.get_transition_probability(state1, state2) *
                self.get_emission_probability(state2, word))

    # Deprecated
    @staticmethod
    def _calculate_transition2(future_scnt, next_future_scnt, curr_next_future_scnt,
                               total_num_scnt, next_scnt, curr_next_scnt):
        """
        Uses deleted interpolation technique to estimate the transition probabilities
        as given limited data, many trigram transitions would not be observed

        Arguments:
        future_scnt -- number of future_state observed
        next_future_scnt -- number of next -> future transitions
        curr_next_future_scnt -- number of curr -> next -> future transitions
        total_num_scnt -- total number of state occurences
        next_scnt -- number of next_state observed
        curr_next_scnt -- number of curr -> next transitions
        """
        n1, n2, n3 = future_scnt, next_future_scnt, curr_next_future_scnt
        c0, c1, c2 = total_num_scnt, next_scnt, curr_next_scnt
        k2 = (math.log(n2 + 1) + 1) / float((math.log(n2 + 1) + 2))
        k3 = (math.log(n3 + 1) + 1) / float((math.log(n3 + 1) + 2))
        res = (
            (k3 * n3 / float(c2))
            + ((1 - k3) * k2 * n2 / float(c1))
            + ((1 - k3) * (1 - k2) * n1 / float(c0))
        )
        return res

    # Deprecated
    def estimate_transition2(self, observations):
        """
        Estimate the second-order transition probabilities given observations

        Arguments:
        observations -- training data, an array of deque with
                        word-state tuples

        Returns:
        transition_probs -- dictionary of second-order transition probabilities
        """
        # Count number of state occurences for use in deleted interpolation
        state_cnts = {}
        state_transition2_cnts = {}
        for state1 in [self.end_states[0]] + self.states:
            state_transition2_cnts[state1] = {}
            for state2 in self.states:
                state_transition2_cnts[state1][state2] = {}
        for ws_deque in observations:
            self._check_end_states(ws_deque)
            for i, ws_pair in enumerate(ws_deque):
                if i >= len(ws_deque) - 2:
                    continue
                curr_state = ws_pair[1]
                next_state = ws_deque[i + 1][1]
                future_state = ws_deque[i + 2][1]
                if future_state not in state_transition2_cnts[curr_state][next_state]:
                    state_transition2_cnts[curr_state][next_state][future_state] = 0
                state_transition2_cnts[curr_state][next_state][future_state] += 1
                # State occurence counts
                if curr_state not in state_cnts:
                    state_cnts[curr_state] = 0
                state_cnts[curr_state] += 1
                if i == len(ws_deque) - 3:
                    for state in [next_state, future_state]:
                        if state not in state_cnts:
                            state_cnts[state] = 0
                        state_cnts[state] += 1
        total_num_states = sum(state_cnts.values())
        transition_probs2 = copy.deepcopy(state_transition2_cnts)
        for curr_state, next_state_transitions in state_transition2_cnts.items():
            for next_state, next_future_scnts in next_state_transitions.items():
                # Calculate transition MLE for each trigram
                for future_state, curr_next_future_scnt in next_future_scnts.items():
                    next_scnt = state_cnts[next_state]
                    future_scnt = state_cnts[future_state]
                    curr_next_scnt = self._transition_cnts[curr_state][next_state]
                    next_future_scnt = self._transition_cnts[next_state][future_state]
                    transition_probs2[curr_state][next_state][future_state] = \
                        HMM2._calculate_transition2(
                            future_scnt, next_future_scnt, curr_next_future_scnt,
                            total_num_states, next_scnt, curr_next_scnt
                        )
        return transition_probs2

    def get_transition2_probability(self, state1, state2, state3):
        """
        Helper function to get transition probability from
        state1 to state2 to state3
        """
        # return self.transition_probs2[state1][state2].get(state3, 0)
        return (self.get_transition_probability(state1, state2)
                * self.get_transition_probability(state2, state3))

    def _dp_helper(self, iteration, curr_state, sequence, viterbi_graph,
                   scaling_constant):
        """
        Dynamic programming helper function for Viterbi algorithm,
        modifies viterbi_graph

        Arguments:
        iteration -- current iteration
        curr_state -- current state
        sequence -- deque with word-state tuples, but with empty states
                    except START and STOP
        viterbi_graph -- dictionary representing incomplete Viterbi graph
        scaling_constant -- scaling constant to avoid potential
                            arithmetic underflow
        """
        start_state = self.end_states[0]
        stop_state = self.end_states[1]
        if iteration < 1 or iteration > len(sequence):
            raise Exception(f"Invalid iteration: {iteration}")
        if curr_state not in self.states + [stop_state]:
            raise Exception(f"Invalid state: {curr_state}")
        if iteration < len(sequence) - 1:
            word = sequence[iteration][0].lower()
        else:
            word = None
        word = self._process_unknown_word(word)
        if iteration == 1:
            # Base case
            alpha = self.get_transition_probability(start_state, curr_state)
            beta = self.get_emission_probability(curr_state, word)
            viterbi_graph[iteration][curr_state] = (
                start_state, alpha * beta * scaling_constant)
            return
        if iteration == 2:
            # Second case
            for prev_state in self.states:
                prev_optimal_prob = viterbi_graph[1][prev_state][1]
                alpha = self.get_transition_probability(prev_state, curr_state)
                beta = self.get_emission_probability(curr_state, word)
                prod = prev_optimal_prob * alpha * beta * scaling_constant
                viterbi_graph[iteration][(prev_state, curr_state)] = \
                    (start_state, prod)
            return
        # Forward recursion
        temp_nodes = []
        for prev_state in self.states:
            for past_state in self.states:
                prev_optimal_prob = \
                    viterbi_graph[iteration - 1].get((past_state, prev_state), ("", 0))[1]
                alpha = self.get_transition2_probability(past_state, prev_state, curr_state)
                if iteration == len(sequence) - 1:
                    beta = 1
                else:
                    beta = self.get_emission_probability(curr_state, word)
                prod = prev_optimal_prob * alpha * beta * scaling_constant
                temp_nodes.append((past_state, prod))
            viterbi_graph[iteration][(prev_state, curr_state)] = max(
                temp_nodes, key=itemgetter(1))

    def viterbi(self, sequence, scaling_constant=1e2):
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
        for iteration in range(1, len(sequence)):
            if iteration == len(sequence) - 1:
                self._dp_helper(iteration, self.end_states[1], sequence,
                                viterbi_graph, scaling_constant)
                continue
            for state in self.states:
                self._dp_helper(iteration, state, sequence, viterbi_graph,
                                scaling_constant)
        return viterbi_graph


def main():
    """Train and test"""
    en_states = [
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
    other_states = [
        "START", "STOP",
        "B-positive", "I-positive",
        "B-neutral", "I-neutral",
        "B-negative", "I-negative",
        "O"
    ]
    data_folders = ["EN", "FR"]
    for folder in data_folders:
        print(f"Training and testing for {folder}...")
        print("=============================================")
        states = en_states if folder == "EN" else other_states
        hmm = HMM2(states)
        train_file = os.path.join(folder, "train")
        test_file = os.path.join(folder, "dev.in")
        gold_file = os.path.join(folder, "dev.out")
        output_file = os.path.join(folder, "dev.test.out")
        # Train and predict
        hmm.train(train_file)
        hmm.predict(test_file, decoding_type="viterbi")
        # Evalute result
        cmd = f"python eval_result.py {gold_file} {output_file}"
        result = subprocess.run(cmd, shell=True, check=True, stdout=subprocess.PIPE)
        text = result.stdout.decode()
        print(text)

if __name__ == "__main__":
    main()
