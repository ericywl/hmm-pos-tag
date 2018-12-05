import os
import subprocess
from collections import deque
from operator import itemgetter
from hmm import HMM

other_entity_states = ["B", "I", "O"]

other_sentiment_states = ["positive","neutral","negative"]

def split_tag(in_filename, out_filename=None, type=None):
    if type != "entity" and type != "sentiment":
        raise Exception("Wrong request of type, put either entity or sentiment")
    dir_path = os.path.dirname(os.path.realpath(in_filename))
    if not out_filename:
        if type == "entity":
            out_filename = os.path.join(dir_path, "dev.part5.entity.out")
        else:
            out_filename = os.path.join(dir_path, "dev.part5.sentiment.out")
    with open(in_filename, "r") as infile:
        with open(out_filename, "w") as outfile:
            for inline in infile:
                if inline == "\n":
                    outfile.write(inline)
                    continue

                line = inline.strip("\n").split(" ")
                if len(line) < 2:
                    pass

                else:
                    if line[1] == "O":
                        entity = "O"
                        sentiment = "none"
                    else:
                        entity, sentiment = line[1].split("-")

                    if type == "entity":
                        line[1] = entity
                    else:
                        line[1] = sentiment

                string = " ".join(line) + "\n"
                outfile.write(string)

def split_entity(in_filename, out_filename=None):
    split_tag(in_filename, out_filename=out_filename, type="entity")

def split_sentiment(in_filename, out_filename=None):
    split_tag(in_filename, out_filename=out_filename, type="sentiment")

def merge(entity_file, sentiment_file, out_filename=None):
    dir_path = os.path.dirname(os.path.realpath(entity_file))
    if not out_filename:
        out_filename = os.path.join(dir_path, "dev.part55.test.out")

    entity_string = []
    with open(entity_file, "r") as en_infile:
        with open(sentiment_file, "r") as sn_infile:
            with open(out_filename, "w") as outfile:
                en_lines = en_infile.readlines()
                sn_lines = sn_infile.readlines()


                if len(en_lines) != len(sn_lines):
                    print("sn_line: " + str(len(sn_lines)))
                    print("em_line: " + str(len(en_lines)))
                    raise Exception("Entity file and sentiment file does not match")


                for i in range(len(en_lines)):
                    if en_lines[i] == "\n" and sn_lines[i] == "\n":
                        outfile.write(en_lines[i])
                        continue

                    en_line = en_lines[i].strip("\n").split(" ")
                    sn_line = sn_lines[i].strip("\n").split(" ")

                    out_line = [en_line[0]]

                    if en_line[1] == "O":
                        out_line.append(en_line[1])
                    else:
                        if sn_line[1] == "none":
                            out_line.append("-".join([en_line[1], "neutral"]))
                        else:
                            out_line.append("-".join([en_line[1], sn_line[1]]))

                    string = " ".join(out_line) + "\n"
                    outfile.write(string)




FR_entity_states = ["START", "STOP",
                       "B", "I", "O"]

FR_sentiment_states = ["START", "STOP",
                          "positive",
                          "neutral",
                          "negative", "none"]
EN_entity_states = [
        "START", "STOP",
        "B", "I", "O"]

EN_sentiment_states = [   "START", "STOP",
        "VP",
        "NP",
        "PP",
        "INTJ",
        "ADJP",
        "SBAR",
        "ADVP",
        "CONJP",
        "O", "PRT", "none"]

folders = ["FR5", "EN5"]

for folder in folders:
    print(f"\nTraining and testing for {folder}...")
    print("=============================================")

    if folder == "FR5":
        entity_states = FR_entity_states
        sentiment_states = FR_sentiment_states
    else:
        entity_states = EN_entity_states
        sentiment_states = EN_sentiment_states

    test_file = os.path.join(folder, "dev.in")
    train_file = os.path.join(folder, "train")
    gold_file = os.path.join(folder, "dev.out")
    en_train = os.path.join(folder, "part5.entity")
    sn_train = os.path.join(folder, "part5.sentiment")
    en_test = os.path.join(folder, "part5.entity.test.out")
    sn_test = os.path.join(folder, "part5.sentiment.test.out")
    output_file = os.path.join(folder, "part5.test.out")

    split_entity(train_file, out_filename=en_train)
    en_hmm = HMM(entity_states)
    en_hmm.train(en_train)
    en_hmm.predict(test_file, out_filename=en_test)

    split_sentiment(train_file, out_filename=sn_train)
    sn_hmm = HMM(sentiment_states)
    sn_hmm.train(sn_train)
    sn_hmm.predict(test_file, out_filename=sn_test)

    merge(en_test, sn_test, out_filename=output_file)

    cmd = f"python eval_result.py {gold_file} {output_file}"
    subprocess.run(cmd, shell=True, check=True)



#FR
# #Entity in gold data: 238
# #Entity in prediction: 462
#
# #Correct Entity : 110
# Entity  precision: 0.2381
# Entity  recall: 0.4622
# Entity  F: 0.3143
#
# #Correct Entity Type : 47
# Entity Type  precision: 0.1017
# Entity Type  recall: 0.1975
# Entity Type  F: 0.1343

#EN
# #Entity in gold data: 802
# #Entity in prediction: 1025
#
# #Correct Entity : 584
# Entity  precision: 0.5698
# Entity  recall: 0.7282
# Entity  F: 0.6393
#
# #Correct Entity Type : 473
# Entity Type  precision: 0.4615
# Entity Type  recall: 0.5898
# Entity Type  F: 0.5178
#
# Process finished with exit code 0