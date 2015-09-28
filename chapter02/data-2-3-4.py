__author__ = 'ryu'

import csv

with open('files/Auto.csv', 'rb') as csvfile:
    spamreader = csv.reader(csvfile)
    for row in spamreader:
        print row