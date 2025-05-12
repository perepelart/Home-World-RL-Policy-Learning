# Home-World-Policy-Learning

## Project Description.

This project was done as a part of the course [MITx 6.86x Machine Learning with Python-From Linear Models to Deep Learning](https://www.edx.org/learn/machine-learning/massachusetts-institute-of-technology-machine-learning-with-python-from-linear-models-to-deep-learning).

We are applying reinforcement learning (RL) to teach agents how to play text-based games. These games rely entirely on text for interaction—players read descriptions and type commands, meaning the true state is only partially observed.

Our experiments use a simple "Home World," a house environment where agents complete text quests (e.g., "You are hungry") by interacting with objects (like eating an apple in the kitchen). The agent receives text descriptions, issues commands, and gets rewards.

Using RL, we aim to develop an agent that learns optimal strategies from these rewards. A core problem is representing the text descriptions as vectors. While assigning unique IDs is simple, it doesn't scale well, leading us to investigate methods like bag-of-words.

The mathematical model for this framework is described in the file [Home World Game Framework.md](https://github.com/perepelart/Home-World-Policy-Learning/blob/167907f70361518aff5ff762558ba9d9dd871c3b/Home%20World%20Game%20Framework.md)

This project requires to complete the following tasks:

1. Implement the <b> tabular Q-learning algorithm </b> for a simple setting where each text description is associated with a unique index.

2. Implement the <b> Q-learning algorithm with linear approximation architecture </b>, using bag-of-words representation for textual state description.

3. Implement a <b>deep Q-network</b>.

4. Use Q-learning algorithms on the Home World game.

---

## Project Structure

This project simulates a decision-making agent in the **Home World** environment using various reinforcement learning algorithms. Below is a description of the key components:

### Agents

* **`agent_dqn.py`**
  Implementation of a **Deep Q-Network (DQN)** using a neural network to approximate the Q-function.

* **`agent_linear.py`**
  Implements **Q-learning** with a **linear function approximator**.

* **`agent_tabular.py`**
  Standard **tabular Q-learning** agent that maintains a Q-table for all state-action pairs.

Note that each agent is fully self-contained, i.e., can be trained and evaluated independently when run.

### Environment

* **`framework.py`**
  Core simulator for the **Home World** environment, defining state transitions, rewards, and dynamics.

* **`game.tsv`**
  Example game instructions that define quests and room descriptions used during simulation.

### Main Logic & Utilities

* **`main.py`**
  Entry point of the project — runs training and evaluation experiments using different agents.

* **`utils.py`**
  Auxiliary functions used across agents and the framework (e.g., data formatting).

### Results

Experimental results can be found in [the following file](results/summary.md).
