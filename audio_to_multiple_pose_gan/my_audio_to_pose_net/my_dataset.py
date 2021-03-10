import numpy
import pandas as pd

#metadata is a dict

class BaseDataset:
    def __init__(self):
        self.metadata_list = None

    def load_element(self):
        pass

    # load the training df for a given speaker
    # and create a generator to iterate through row objects
    def load_training_df(self, train_csv, speaker):
        df = pd.read_csv(train_csv)
        df = df[df['speaker'] == speaker]
        df = df[df['dataset'] == 'train']
        return df