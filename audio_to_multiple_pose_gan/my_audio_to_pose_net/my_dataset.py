import numpy


#metadata is a dict

class BaseDataset:
    def __init__(self):
        self.metadata_list = None

    def load_element(self):
        pass