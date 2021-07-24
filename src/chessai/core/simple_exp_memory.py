
import numpy as np


class SimpleBufferedExperienceMemory():
    """
    An implementation of a simple buffered experience memory used for reinforcement learning.
    It supports following actions and use cases:

    1) applying a single experience tuple to the buffer
    2) sampling randomized batches from the buffer (post-process. function can be customized)
    3) clearing the buffer
    """

    def __init__(self, mem_size: int=100000, min_exp_size: int=10000,
                 batch_size: int=128, batch_transform_func=None):
        super(SimpleBufferedExperienceMemory, self).__init__()

        # assign memory hyper-parameters
        self.mem_size = mem_size
        self.min_exp_size = min_exp_size
        self.batch_size = batch_size

        # initialize the batch postprocessing function
        self.batch_transform_func = batch_transform_func

        # initialize an empty memory
        self.memory = list()


    def add_experience(self, exp: tuple):

        # append the experience to the buffer
        self.memory.append(exp)

        # forget the oldest experience in the buffer (if the buffer overflows)
        if len(self.memory) > self.mem_size: self.memory.pop(0)


    def clear_buffer(self):
        self.memory = list()
        # TODO: think of adding proper garbage collection here


    def sample_exp_batch(self) -> tuple:

        # make sure there are enough experiences to sample a batch from
        if len(self.memory) < self.min_exp_size: return None

        # sample random indices to be drawn from the buffer
        indices = np.random.randint(low=0, high=len(self.memory),
                                    size=self.batch_size, dtype=np.int32)
        sampled_batch = [self.memory[i] for i in indices]

        # format the sampled batch properly
        return sampled_batch if self.batch_transform_func is None \
            else self.batch_transform_func(sampled_batch)
