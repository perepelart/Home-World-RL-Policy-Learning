import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import framework
import utils

class TabularQLAgent:
    """Tabular Q-Learning Agent implementation."""

    def __init__(self,
                 gamma = 0.5,
                 training_ep = 0.5,
                 testing_ep = 0.05,
                 num_epochs = 200,
                 num_epis_train = 25,
                 num_epis_test = 50,
                 alpha = 0.1):
        """
        Initializes the Tabular Q-Learning agent.

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

        # These will be initialized in the run method
        self.q_func = None
        self.dict_room_desc = None
        self.dict_quest_desc = None
        self.num_room_desc = None
        self.num_quests = None

    def _epsilon_greedy(self, state_1, state_2, epsilon):
        """
        Selects an action using epsilon-greedy policy.

        Args:
            state_1 (int): Index of the current room description.
            state_2 (int): Index of the current quest description.
            epsilon (float): Probability of choosing a random command.

        Returns:
            (int, int): Indices of the selected action and object.
        """
        if np.random.rand() < epsilon:
            # Explore: choose random action/object
            action_index = np.random.choice(self.num_actions)
            object_index = np.random.choice(self.num_objects)
        else:
            # Exploit: choose the best action/object based on Q-function
            q_slice = self.q_func[state_1, state_2, :, :]
            flat_index = np.argmax(q_slice)
            action_index, object_index = np.unravel_index(flat_index, q_slice.shape)
        return action_index, object_index

    def _tabular_q_learning(self, current_state_1, current_state_2, action_index,
                            object_index, reward, next_state_1, next_state_2,
                            terminal):
        """
        Updates the Q-function for a given transition.

        Args:
            current_state_1, current_state_2 (int): Indices of the current state.
            action_index (int): Index of the action taken.
            object_index (int): Index of the object used.
            reward (float): Immediate reward received.
            next_state_1, next_state_2 (int): Indices of the next state.
            terminal (bool): Whether the episode has ended.
        """
        current_q = self.q_func[current_state_1, current_state_2, action_index, object_index]
        
        # Target Q-value calculation
        max_next_q = 0
        if not terminal:
            max_next_q = np.max(self.q_func[next_state_1, next_state_2, :, :])
            
        target_q = reward + self.gamma * max_next_q

        # Q-value update
        self.q_func[current_state_1, current_state_2, action_index, object_index] = \
            (1 - self.alpha) * current_q + self.alpha * target_q


    def _run_episode(self, for_training):
        """
        Runs a single episode.

        Args:
            for_training (bool): If True, updates Q-function. Otherwise, calculates reward.

        Returns:
            float or None: Cumulative discounted reward if not training, else None.
        """
        epsilon = self.training_ep if for_training else self.testing_ep
        rewards_list = []
        (current_room_desc, current_quest_desc, terminal) = framework.newGame()

        while not terminal:
            current_room_index = self.dict_room_desc[current_room_desc]
            current_quest_index = self.dict_quest_desc[current_quest_desc]

            # Choose action
            action_index, object_index = self._epsilon_greedy(
                current_room_index, current_quest_index, epsilon
            )

            # Take step
            next_room_desc, next_quest_desc, reward, terminal = framework.step_game(
                current_room_desc, current_quest_desc, action_index, object_index
            )
            next_room_index = self.dict_room_desc[next_room_desc]
            next_quest_index = self.dict_quest_desc[next_quest_desc]

            if for_training:
                # Update Q-function
                self._tabular_q_learning(
                    current_room_index, current_quest_index,
                    action_index, object_index,
                    reward,
                    next_room_index, next_quest_index,
                    terminal
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

    def run(self, num_room_desc, num_quests, dict_room_desc, dict_quest_desc):
        """
        Runs the full training and testing process for the agent.

        Args:
            num_room_desc (int): Total number of unique room descriptions.
            num_quests (int): Total number of unique quest descriptions.
            dict_room_desc (dict): Mapping from room description string to index.
            dict_quest_desc (dict): Mapping from quest description string to index.

        Returns:
            np.ndarray: Array of average test rewards per epoch for this run.
        """
        # Initialize Q-function and store state dictionaries
        self.num_room_desc = num_room_desc
        self.num_quests = num_quests
        self.dict_room_desc = dict_room_desc
        self.dict_quest_desc = dict_quest_desc
        self.q_func = np.zeros((self.num_room_desc, self.num_quests,
                                self.num_actions, self.num_objects))

        epoch_rewards_test = []
        print("Running Tabular Q-Learning...")
        pbar = tqdm(range(self.num_epochs), ncols=80, desc="TabularQL")
        for epoch in pbar:
            avg_reward = self._run_epoch()
            epoch_rewards_test.append(avg_reward)
            if epoch_rewards_test: # Avoid error on first iteration
                 pbar.set_description(
                    f"TabularQL | Avg reward: {np.mean(epoch_rewards_test):.4f} | "
                    f"EWMA reward: {utils.ewma(epoch_rewards_test):.4f}"
                )

        return np.array(epoch_rewards_test)


if __name__ == '__main__':
    # Constants for standalone testing
    NUM_RUNS_MAIN = 5

    # Load game data and create state dictionaries
    framework.load_game_data()
    (dict_room_desc_main, dict_quest_desc_main) = framework.make_all_states_index()
    num_room_desc_main = len(dict_room_desc_main)
    num_quests_main = len(dict_quest_desc_main)

    all_runs_rewards = []
    for i in range(NUM_RUNS_MAIN):
        print(f"\n--- Starting Run {i+1}/{NUM_RUNS_MAIN} ---")
        # Instantiate the agent
        agent = TabularQLAgent(num_epochs=200, alpha=1e-10) # Example params

        # Run the agent
        run_rewards = agent.run(
            num_room_desc_main,
            num_quests_main,
            dict_room_desc_main,
            dict_quest_desc_main
        )
        all_runs_rewards.append(run_rewards)

    # Plotting results
    all_runs_rewards = np.array(all_runs_rewards)
    mean_rewards = np.mean(all_runs_rewards, axis=0)
    
    plt.figure()
    plt.plot(np.arange(len(mean_rewards)), mean_rewards)
    plt.xlabel('Epochs')
    plt.ylabel('Average Test Reward')
    plt.title(f'Tabular QL Performance (Avg over {NUM_RUNS_MAIN} runs)')
    plt.grid(True)
    plt.show()
