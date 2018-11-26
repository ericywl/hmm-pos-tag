#!/usr/bin/env python3
"""ML Design Project"""

from collections import deque

class HMM:
    """HMM class"""

    def __init__(self, states):
        self.states = states
        self.emission_probabilities = {}
        self.transition_probabilities = {}

    def calculate_emission(self, observations):
        temp_emission = {}
        for state in self.states:
            temp_emission[state] = {}
        for ws_deque in observations:
            for word, state in ws_deque:
                if state in ["START", "STOP"]:
                    continue
                if word not in temp_emission[state]:
                    temp_emission[state][word] = 0
                temp_emission[state][word] += 1


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
    print(process_file("EN/dev.in", "test"))
