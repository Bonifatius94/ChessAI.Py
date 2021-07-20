
import numpy as np


class StackedNumpyBatchFormatter():

    def __init__(self, out_types: tuple):
        super(StackedNumpyBatchFormatter, self).__init__()
        self.out_types = out_types
        # TODO: think of adding some reshaping functionality as well


    def format_batch(self, batch: list):

        # make sure the given batch is not None
        if batch is None: raise ValueError('Invalid argument! Batch must not be None!')

        # make sure the batch is not empty, otherwise abort with empty tuple
        batch_len = len(batch)
        if batch_len == 0: return tuple()

        # determine the amount of attributes (assuming each
        # batch item has the same amount of attributes)
        num_attr = len(batch[0])

        # sample the attributes from each tuple on the batch
        # -> one stacked numpy array per attribute
        batch = [[x[attr] for x in batch] for attr in num_attr]
        batch = [np.array(batch[attr], dtype=self.out_types[attr]) for attr in num_attr]

        # return the stacked batch attributes as a tuple
        return tuple(batch)