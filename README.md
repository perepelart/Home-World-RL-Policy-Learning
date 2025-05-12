# Home-World-Policy-Learning

## Project Description.

This project was done as a part of the course [MITx 6.86x Machine Learning with Python-From Linear Models to Deep Learning](https://www.edx.org/learn/machine-learning/massachusetts-institute-of-technology-machine-learning-with-python-from-linear-models-to-deep-learning).

In this project, we address the task of learning control policies for text-based games using reinforcement learning. In these games, all interactions between players and the virtual world are through text. The current world state is described by elaborate text, and the underlying state is not directly observable. Players read descriptions of the state and respond with natural language commands to take actions.

For this project we will conduct experiments on a small Home World, which mimic the environment of a typical house. The world consists of a few rooms, and each room contains a representative object that the player can interact with. For instance, the kitchen has an apple that the player can eat. The goal of the player is to finish some quest. An example of a quest given to the player in text is *You are hungry now*. To complete this quest, the player has to navigate through the house to reach the kitchen and eat the apple. In this game, the room is hidden from the player, who only receives a description of the underlying room. At each step, the player read the text describing the current room and the quest, and respond with some command (e.g., eat apple ). The player then receives some reward that depends on the state and given command.

In order to design an autonomous game player, we will employ a reinforcement learning framework to learn command policies using game rewards as feedback. Since the state observable to the player is described in text, we have to choose a mechanism that maps text descriptions into vector representations. A naive approach is to create a map that assigns a unique index for each text description. However, such approach becomes difficult to implement when the number of textual state descriptions are huge. An alternative method is to use a bag-of-words representation derived from the text description. 

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
