import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import framework
import utils

class LinearQLAgent:
    """Linear Q-Learning Agent implementation using feature vectors."""

    def __init__(self,
                 gamma=0.5,
                 training_ep=0.5,
                 testing_ep=0.05,
                 num_epochs=600,
                 num_epis_train=25,
                 num_epis_test=50,
                 alpha=0.01):
        """
        Initializes the Linear Q-Learning agent.

        Args:
            gamma (float): Discount factor.
            training_ep (float): Epsilon for epsilon-greedy policy during training.
            testing_ep (float): Epsilon for epsilon-greedy policy during testing.
            num_epochs (int): Number of training epochs.
            num_epis_train (int): Number of training episodes per epoch.
            num_epis_test (int): Number of testing episodes per epoch.
            alpha (float): Learning rate.
        """
        self.gamma = gamma
        self.training_ep = training_ep
        self.testing_ep = testing_ep
        self.num_epochs = num_epochs
        self.num_epis_train = num_epis_train
        self.num_epis_test = num_epis_test
        self.alpha = alpha

        self.actions = framework.get_actions()
        self.objects = framework.get_objects()
        self.num_actions = len(self.actions)
        self.num_objects = len(self.objects)
        self.action_dim = self.num_actions * self.num_objects

        # These will be initialized in the run method
        self.theta = None
        self.dictionary = None
        self.state_dim = None

    def _tuple2index(self, action_index, object_index):
        """Converts (action_index, object_index) tuple to a flat index."""
        return action_index * self.num_objects + object_index

    def _index2tuple(self, index):
        """Converts a flat index back to (action_index, object_index) tuple."""
        return index // self.num_objects, index % self.num_objects

    def _epsilon_greedy(self, state_vector, epsilon):
        """
        Selects an action using epsilon-greedy policy with linear approximation.

        Args:
            state_vector (np.ndarray): Feature vector of the current state.
            epsilon (float): Probability of choosing a random command.

        Returns:
            (int, int): Indices of the selected action and object.
        """
        if np.random.rand() < epsilon:
            # Explore: choose random action/object
            action_index = np.random.choice(self.num_actions)
            object_index = np.random.choice(self.num_objects)
        else:
            # Exploit: choose the best action/object based on Q(s, a) = theta * phi(s)
            q_values = self.theta @ state_vector # Shape: (action_dim,)
            best_flat_index = np.argmax(q_values)
            action_index, object_index = self._index2tuple(best_flat_index)
        return action_index, object_index

    def _linear_q_learning(self, current_state_vector, action_index, object_index,
                           reward, next_state_vector, terminal):
        """
        Updates the weight matrix theta using linear Q-learning update rule.

        Args:
            current_state_vector (np.ndarray): Feature vector of the current state.
            action_index (int): Index of the action taken.
            object_index (int): Index of the object used.
            reward (float): Immediate reward received.
            next_state_vector (np.ndarray): Feature vector of the next state.
            terminal (bool): Whether the episode has ended.
        """
        action_obj_index = self._tuple2index(action_index, object_index)

        # Calculate current Q-value estimate: Q(s, a) = theta_a * phi(s)
        q_s_a = self.theta[action_obj_index] @ current_state_vector

        # Calculate target Q-value
        max_next_q = 0
        if not terminal:
            q_next_state = self.theta @ next_state_vector # Q values for all actions in next state
            max_next_q = np.max(q_next_state)

        target_q = reward + self.gamma * max_next_q
        
        # Calculate TD error
        td_error = target_q - q_s_a

        # Update theta for the specific action-object pair
        # Gradient is td_error * phi(s)
        self.theta[action_obj_index] += self.alpha * td_error * current_state_vector


    def _run_episode(self, for_training):
        """
        Runs a single episode.

        Args:
            for_training (bool): If True, updates theta. Otherwise, calculates reward.

        Returns:
            float or None: Cumulative discounted reward if not training, else None.
        """
        epsilon = self.training_ep if for_training else self.testing_ep
        rewards_list = []
        (current_room_desc, current_quest_desc, terminal) = framework.newGame()

        while not terminal:
            # Get state vector
            current_state_text = current_room_desc + current_quest_desc
            current_state_vector = utils.extract_bow_feature_vector(
                current_state_text, self.dictionary
            )

            # Choose action
            action_index, object_index = self._epsilon_greedy(current_state_vector, epsilon)

            # Take step
            next_room_desc, next_quest_desc, reward, terminal = framework.step_game(
                current_room_desc, current_quest_desc, action_index, object_index
            )

            # Get next state vector
            next_state_text = next_room_desc + next_quest_desc
            next_state_vector = utils.extract_bow_feature_vector(
                next_state_text, self.dictionary
            )

            if for_training:
                # Update weights
                self._linear_q_learning(
                    current_state_vector, action_index, object_index,
                    reward, next_state_vector, terminal
                )
            else:
                # Collect reward for testing
                rewards_list.append(reward)

            # Move to next state
            current_room_desc = next_room_desc
            current_quest_desc = next_quest_desc

        if not for_training:
            # Calculate cumulative discounted reward
            discounts = np.array([self.gamma**t for t in range(len(rewards_list))])
            epi_reward = np.dot(discounts, np.array(rewards_list))
            return epi_reward

        return None # No reward returned during training episode

    def _run_epoch(self):
        """Runs one epoch of training and testing."""
         # Training phase
        for _ in range(self.num_epis_train):
            self._run_episode(for_training=True)

        # Testing phase
        test_rewards = []
        for _ in range(self.num_epis_test):
            reward = self._run_episode(for_training=False)
            test_rewards.append(reward)

        return np.mean(np.array(test_rewards))

    def run(self, dictionary, state_dim):
        """
        Runs the full training and testing process for the agent.

        Args:
            dictionary (dict): Bag-of-words dictionary mapping words to indices.
            state_dim (int): Dimension of the state feature vector (size of dictionary).

        Returns:
            np.ndarray: Array of average test rewards per epoch for this run.
        """
        # Initialize weight matrix theta and store dictionary info
        self.dictionary = dictionary
        self.state_dim = state_dim
        self.theta = np.zeros((self.action_dim, self.state_dim))

        epoch_rewards_test = []
        print("Running Linear Q-Learning...")
        pbar = tqdm(range(self.num_epochs), ncols=80, desc="LinearQL")
        for epoch in pbar:
            avg_reward = self._run_epoch()
            epoch_rewards_test.append(avg_reward)
            if epoch_rewards_test:
                pbar.set_description(
                     f"LinearQL | Avg reward: {np.mean(epoch_rewards_test):.4f} | "
                     f"EWMA reward: {utils.ewma(epoch_rewards_test):.4f}"
                 )

        return np.array(epoch_rewards_test)


if __name__ == '__main__':
    # Constants for standalone testing
    NUM_RUNS_MAIN = 5 # Use a different name

    # Load data and create dictionary
    state_texts_main = utils.load_data('game.tsv')
    dictionary_main = utils.bag_of_words(state_texts_main)
    state_dim_main = len(dictionary_main)

    # Load game data
    framework.load_game_data()

    all_runs_rewards = []
    for i in range(NUM_RUNS_MAIN):
        print(f"\n--- Starting Run {i+1}/{NUM_RUNS_MAIN} ---")
        # Instantiate the agent
        agent = LinearQLAgent(num_epochs=600, alpha=0.01) # Example params

        # Run the agent
        run_rewards = agent.run(dictionary_main, state_dim_main)
        all_runs_rewards.append(run_rewards)

    # Plotting results
    all_runs_rewards = np.array(all_runs_rewards)
    mean_rewards = np.mean(all_runs_rewards, axis=0)

    plt.figure()
    plt.plot(np.arange(len(mean_rewards)), mean_rewards)
    plt.xlabel('Epochs')
    plt.ylabel('Average Test Reward')
    plt.title(f'Linear QL Performance (Avg over {NUM_RUNS_MAIN} runs)')
    plt.grid(True)
    plt.show()