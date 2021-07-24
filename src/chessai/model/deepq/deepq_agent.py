
import numpy as np

from .deepq_model import ChessDeepQModel
from chessai.core import ChessGameEnv, SimpleBufferedExperienceMemory
from chessai.core import StackedNumpyBatchFormatter, AbstractModelAdjuster
from chessai.dataset import convert_states


class ChessDeepQAgent():
    """
    This is an implementation of a Deep-Q agent playing chess.
    Given a game context it can choose 'best' action and perform training steps to maximize rewards.
    """

    def __init__(self, env: ChessGameEnv, model_adj: AbstractModelAdjuster,
                 epsilon: float=0.1, batch_size: int=32):
        super(ChessDeepQAgent, self).__init__()

        # assign overloaded hyper-params
        self.env = env
        self.epsilon = epsilon

        # create a value estimation model
        self.model = ChessDeepQModel()
        self.model_adj = model_adj

        # initialize a buffered experience memory
        batch_types = (np.float32, np.int32, np.float32, np.float32, np.float32)
        self.batch_formatter = StackedNumpyBatchFormatter(batch_types)
        self.exp_memory = SimpleBufferedExperienceMemory(batch_size=batch_size,
            batch_transform_func=self.batch_formatter.format_batch)

        # # add formatting functions detecting raw chessboards and formatting them properly
        # is_raw_chessboard = lambda attr: isinstance(attr, np.ndarray) \
        #     and len(attr.shape) == 2 and attr.shape[1] == 13 and attr.dtype == np.uint64
        # conv_batch_attr = lambda attr: convert_states(attr) if is_raw_chessboard(attr) else attr
        # format_states = lambda batch: tuple([conv_batch_attr(attr) for attr in batch])

        # # apply the batch formatting logic to the experience memory
        # batch_transform_func = lambda batch: format_states(self.batch_formatter.format_batch(batch))
        # self.exp_memory = SimpleBufferedExperienceMemory(batch_transform_func=batch_transform_func)


    def choose_action(self) -> int:

        # generate all possible actions
        poss_actions = self.env.get_possible_actions()

        # define functions to select either an exploring or an exploiting action
        get_explore_action = lambda: np.random.choice(poss_actions, size=1).item()
        get_poss_next_states = lambda a: convert_states(np.array([self.env.simulate_action(x) for x in a]))
        get_action_est_values = lambda a: self.model(get_poss_next_states(a)).numpy()
        get_exploit_action = lambda a: poss_actions[np.squeeze(np.argmax(get_action_est_values(a))).item()]

        # determine the next action (epsilon-greedy policy)
        explore = np.random.uniform() < self.epsilon
        return get_explore_action() if explore else get_exploit_action(poss_actions)
        # TODO: think of adding a proper epsilon decay function


    def train_step(self, experience: tuple):

        # apply experience and sample a training batch
        self.exp_memory.add_experience(experience)
        train_batch = self.exp_memory.sample_exp_batch()

        # if there's a batch to be trained on
        if train_batch is not None:

            # make sure the chess boards are formatted as the model expects them
            train_batch = (convert_states(train_batch[0]), train_batch[1],
                train_batch[2], convert_states(train_batch[3]), train_batch[4])

            # train on the sampled batch and update the model accordingly
            self.model_adj.update_weights(self.model, train_batch)


    # TODO: think of adding functionality for loading / saving the model
