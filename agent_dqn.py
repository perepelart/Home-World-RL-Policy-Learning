import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import framework
import utils


class DQN(nn.Module):
    """Simple deep Q network."""
    def __init__(self, state_dim, action_dim, object_dim, hidden_size=100):
        super(DQN, self).__init__()
        self.state_encoder = nn.Linear(state_dim, hidden_size)
        self.state2action = nn.Linear(hidden_size, action_dim)
        self.state2object = nn.Linear(hidden_size, object_dim)

    def forward(self, x):
        state = F.relu(self.state_encoder(x))
        # Return separate Q-values for actions and objects
        return self.state2action(state), self.state2object(state)


class DeepQLAgent:
    """Deep Q-Learning Agent implementation."""

    def __init__(self,
                 gamma=0.5,
                 training_ep=0.5,
                 testing_ep=0.05,
                 num_epochs=300,
                 num_epis_train=25,
                 num_epis_test=50,
                 alpha=0.001, 
                 hidden_size=100): 
        """
        Initializes the Deep Q-Learning agent.

        Args:
            gamma (float): Discount factor.
            training_ep (float): Epsilon for epsilon-greedy policy during training.
            testing_ep (float): Epsilon for epsilon-greedy policy during testing.
            num_epochs (int): Number of training epochs.
            num_epis_train (int): Number of training episodes per epoch.
            num_epis_test (int): Number of testing episodes per epoch.
            alpha (float): Learning rate for the optimizer.
            hidden_size (int): Size of the hidden layer in the DQN.
        """
        self.gamma = gamma
        self.training_ep = training_ep
        self.testing_ep = testing_ep
        self.num_epochs = num_epochs
        self.num_epis_train = num_epis_train
        self.num_epis_test = num_epis_test
        self.alpha = alpha
        self.hidden_size = hidden_size

        self.actions = framework.get_actions()
        self.objects = framework.get_objects()
        self.num_actions = len(self.actions)
        self.num_objects = len(self.objects)

        # These will be initialized in the run method
        self.model = None
        self.optimizer = None
        self.dictionary = None
        self.state_dim = None

    def _epsilon_greedy(self, state_vector, epsilon):
        """
        Selects an action using epsilon-greedy policy with DQN.

        Args:
            state_vector (torch.FloatTensor): Feature vector of the current state.
            epsilon (float): Probability of choosing a random command.

        Returns:
            (int, int): Indices of the selected action and object (as Python ints).
        """
        if np.random.rand() < epsilon:
            # Explore
            action_index = np.random.choice(self.num_actions)
            object_index = np.random.choice(self.num_objects)
        else:
            # Exploit
            self.model.eval() # Set model to evaluation mode
            with torch.no_grad(): # Disable gradient calculation for inference
                 q_actions, q_objects = self.model(state_vector)
            self.model.train() # Set model back to training mode
            # Get the index of the max Q-value
            action_index = torch.argmax(q_actions).item() # .item() converts tensor to Python int
            object_index = torch.argmax(q_objects).item()
        return action_index, object_index

    def _deep_q_learning(self, current_state_vector, action_index, object_index,
                         reward, next_state_vector, terminal):
        """
        Performs a single update step on the DQN weights.

        Args:
            current_state_vector (torch.FloatTensor): Feature vector of the current state.
            action_index (int): Index of the action taken.
            object_index (int): Index of the object used.
            reward (float): Immediate reward received.
            next_state_vector (torch.FloatTensor): Feature vector of the next state.
            terminal (bool): Whether the episode has ended.
        """
        # Predict Q-values for the current state
        q_actions_current, q_objects_current = self.model(current_state_vector)

        # Get the specific Q-values for the action/object taken
        q_action_taken = q_actions_current[action_index]
        q_object_taken = q_objects_current[object_index]
        # Average the Q-values for the taken action-object pair
        current_q = 0.5 * (q_action_taken + q_object_taken)

        # Calculate target Q-value using the next state
        max_next_q = 0.0
        if not terminal:
            self.model.eval() # Evaluation mode for predicting next Q
            with torch.no_grad():
                 q_actions_next, q_objects_next = self.model(next_state_vector)
            self.model.train() # Back to training mode
            # Use the average of the max action Q and max object Q in the next state
            max_next_q = 0.5 * (torch.max(q_actions_next) + torch.max(q_objects_next))

        # Target = reward + gamma * max_next_Q (if not terminal)
        target_q = torch.tensor(reward, dtype=torch.float32) + self.gamma * max_next_q

        # Calculate loss (e.g., Mean Squared Error)
        loss = F.mse_loss(current_q, target_q)

        # Backpropagation
        self.optimizer.zero_grad() # Clear previous gradients
        loss.backward()          # Compute gradients
        self.optimizer.step()        # Update weights


    def _run_episode(self, for_training):
        """
        Runs a single episode.

        Args:
            for_training (bool): If True, updates DQN weights. Otherwise, calculates reward.

        Returns:
            float or None: Cumulative discounted reward if not training, else None.
        """
        epsilon = self.training_ep if for_training else self.testing_ep
        rewards_list = []
        (current_room_desc, current_quest_desc, terminal) = framework.newGame()

        while not terminal:
            # Get state vector
            current_state_text = current_room_desc + current_quest_desc
            # Convert state to FloatTensor for PyTorch
            current_state_vector = torch.FloatTensor(
                utils.extract_bow_feature_vector(current_state_text, self.dictionary)
            )

            # Choose action
            action_index, object_index = self._epsilon_greedy(current_state_vector, epsilon)

            # Take step
            next_room_desc, next_quest_desc, reward, terminal = framework.step_game(
                current_room_desc, current_quest_desc, action_index, object_index
            )

            # Get next state vector
            next_state_text = next_room_desc + next_quest_desc
            next_state_vector = torch.FloatTensor(
                utils.extract_bow_feature_vector(next_state_text, self.dictionary)
            )


            if for_training:
                # Update weights
                self._deep_q_learning(
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
        self.model.train() # Set model to training mode
        for _ in range(self.num_epis_train):
            self._run_episode(for_training=True)

        # Testing phase
        self.model.eval() # Set model to evaluation mode
        test_rewards = []
        with torch.no_grad(): # Disable gradients during testing
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
        # Initialize DQN model, optimizer, and store dictionary info
        self.dictionary = dictionary
        self.state_dim = state_dim
        self.model = DQN(self.state_dim, self.num_actions, self.num_objects, self.hidden_size)
        # self.optimizer = optim.Adam(self.model.parameters(), lr=self.alpha) # Adam worked worse
        self.optimizer = optim.SGD(self.model.parameters(), lr=self.alpha) # Stick with SGD

        epoch_rewards_test = []
        print("Running Deep Q-Learning (DQN)...")
        pbar = tqdm(range(self.num_epochs), ncols=80, desc="DQN")
        for epoch in pbar:
            avg_reward = self._run_epoch()
            epoch_rewards_test.append(avg_reward)
            if epoch_rewards_test:
                pbar.set_description(
                     f"DQN | Avg reward: {np.mean(epoch_rewards_test):.4f} | "
                     f"EWMA reward: {utils.ewma(epoch_rewards_test):.4f}"
                 )

        return np.array(epoch_rewards_test)


if __name__ == '__main__':
    # Constants for standalone testing
    NUM_RUNS_MAIN = 5

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
        agent = DeepQLAgent(num_epochs=300, alpha=0.001, hidden_size=128) # Example params

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
    plt.title(f'DQN QL Performance (Avg over {NUM_RUNS_MAIN} runs)')
    plt.grid(True)
    plt.show()
