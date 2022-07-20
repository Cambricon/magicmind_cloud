import os

class Record:
    def __init__(self, filename):
        self.file = open(filename, "w")

    def write(self, line, _print = False):
        self.file.write(line + "\n")
        if _print:
            print(line)
