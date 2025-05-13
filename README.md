# Home World: RL-Based Policy Learning.

This work was done as a part of the course [MITx 6.86x Machine Learning with Python-From Linear Models to Deep Learning](https://www.edx.org/learn/machine-learning/massachusetts-institute-of-technology-machine-learning-with-python-from-linear-models-to-deep-learning).

## Project Description.

In this project, we use reinforcement learning (RL) to train agents to solve text-based games by learning control policies. In such games, all interactions — state descriptions and agent actions — are written in natural language, so the underlying world state is only partially observable through these textual descriptions.

To conduct the experiments, we employ a small "Home World", simulating a house with four connected rooms containing interactive objects. The agent must complete quests presented textually that involve interacting with these objects. We assume that if the agent is already in the room associated with the quest, they can complete it by doing exactly one action. In each cycle, the agent receives a text describing the current state and quest, submits a command, and receives a reward based on the resulting state and the command.

For instance, the *kitchen* contains an *apple* that the agent can eat. The agent may receive a quest such as *You are hungry*, which requires navigating through the house to reach the kitchen and consume the apple. The apple can be consumed directly without additional preparation, such as slicing.

To design an autonomous game player, we leverage a reinforcement learning framework to learn command policies using game rewards as feedback. One challenge is mapping textual descriptions into vector representations. We first test a naive one-hot encoding by assigning a unique index to each text state. However, this approach becomes impractical as the number of distinct descriptions grows. We therefore also explore a bag-of-words representation.

Details of the mathematical model for this framework are provided in [**`Home World Game Framework.md`**](https://github.com/perepelart/Home-World-Policy-Learning/blob/167907f70361518aff5ff762558ba9d9dd871c3b/Home%20World%20Game%20Framework.md)

This project involves completing the following tasks:

1. Implement the <b> tabular Q-learning algorithm </b> using one-hot encoding of text description.

2. Implement the <b> Q-learning algorithm with linear approximation</b>, using a bag-of-words representation for textual state description.

3. Implement a <b>deep Q-network</b>.

4. **Apply** the Q-learning algorithms to the Home World game and **compare** the results.

---

## Project Structure

This project simulates a decision-making agent in the **Home World** environment using various reinforcement learning algorithms. Below is a description of the key components:

### Agents

* [**`agent_dqn.py`**](agent_dqn.py)
  Implementation of a **Deep Q-Network (DQN)** using a neural network to approximate the Q-function.

* [**`agent_linear.py`**](agent_linear.py)
  Implements **Q-learning** with a **linear function approximator**.

* [**`agent_tabular.py`**](agent_tabular.py)
  Standard **tabular Q-learning** agent that maintains a Q-table for all state-action pairs.

Note that each agent is fully self-contained, i.e., can be trained and evaluated independently when run.

### Environment

* [**`framework.py`**](framework.py)
  Core simulator for the **Home World** environment, defining state transitions, rewards, and dynamics.

* [**`game.tsv`**](game.tsv)
  Example game instructions that define quests and room descriptions used during simulation.
  
These two files should be used either with **`main.py`** or one of the agents for their evaluation.

### Main Logic & Utilities

* [**`main.py`**](main.py)
  Entry point of the project — runs training and evaluation experiments using different agents.

* [**`utils.py`**](utils.py)
  Auxiliary functions used across agents and the framework (e.g., data formatting).

### Results

Experimental results can be found in [**`summary.md`**](results/summary.md).
