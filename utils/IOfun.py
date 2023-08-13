# -*- coding: utf-8 -*-
"""
Created on Tue Jun 23 11:57:39 2020

@author: Yuanhao
"""

import traceback
import sys

# Context manager that copies stdout and any exceptions to a log file
class Tee(object):
    def __init__(self, filename):
        self.file = open(filename, 'w')
        self.stdout = sys.stdout

    def __enter__(self):
        sys.stdout = self

    def __exit__(self, exc_type, exc_value, tb):
        sys.stdout = self.stdout
        if exc_type is not None:
            self.file.write(traceback.format_exc())
        self.file.close()

    def write(self, data):
        self.file.write(data)
        self.stdout.write(data)

    def flush(self):
        self.file.flush()
        self.stdout.flush()
        
# print("Print")
# with Tee('test.txt'):
#     print("Print+Write")
#     raise Exception("Test")
# print("Print")