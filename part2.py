#!/usr/bin/env python3
"""Part 2 of ML Design Project"""

from collections import deque

class HMM:

    def __init__(self):
        self.emission_probabilities = {}
        self.transition_probabilities = {}

    def calculate_emission(self, observations):


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
