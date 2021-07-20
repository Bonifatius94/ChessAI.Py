
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

    def __init__(self, env: ChessGameEnv, model_adj: AbstractModelAdjuster, epsilon: float=0.1):
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
        self.exp_memory = SimpleBufferedExperienceMemory(
            batch_transform_func=self.batch_formatter.format_batch)


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

        # train on the sampled batch and update the model accordingly
        if train_batch is not None: self.model_adj.update_weights(self.model, train_batch)


    # TODO: think of adding functionality for loading / saving the model
