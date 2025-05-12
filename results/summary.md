[The image](rewards_by_agent_comparison.png) ([an interactive version](interactive_plot.html) is also available, which should be downloaded and opened in a browser) shows a comparison of the average episodic rewards obtained in the Home World environment by executing policies learned through Standard Tabular Q-learning, Q-learning with a Linear Function Approximaton, and Deep Q-Network (DQN) learning.


| Algorithm                                  | Average Episodic Reward |
|-------------------------------------------|--------------------------|
| Standard Tabular Q-Learning               | $0.49800$                   |
| Q-Learning with Linear Function Approximation | $0.38000$              |
| Deep Q-Network Learning                   | $0.49864$                 |

In this environment, the tabular method achieved performance nearly identical to DQN while requiring less time to converge. In contrast, Q-learning with linear function approximation consistently underperformed relative to both.

Parameters specific to the learning algorithm are listed below, along with common parameters that were kept consistent across experiments. The results presented were averaged over five independent runs for each agent configuration.

| Parameter                   | Tabular QL      | Linear QL           | DQN              |
|----------------------------|------------------|----------------------|------------------|
| **Learning Rate ($\alpha$)**      | $0.1$      | $0.01$                 | $0.1$      |
| **Training Epsilon ($\varepsilon_{\text{train}}$)** | $0.5$              | $0.5$                  | $0.5$              |
| **Number of Epochs**       | $200$              | $600$                  | $300$              |
| **Hidden Layer Size**      | —              | —                  | $128$              |
| **Optimizer**              | —              | —                  | $\mathrm{SGD}$              |
| **Common Parameters**      |                  |                      |                  |
| **Discount Factor ($\gamma$)**    | $0.5$              | $0.5$                  | $0.5$              |
| **Testing Epsilon ($\varepsilon_{\text{test}}$)**  | $0.05$             | $0.05$                 | $0.05$             |
| **Training Episodes/Epoch**| $25$               | $25$                   | $25$               |
| **Testing Episodes/Epoch** | $50$               | $50$                   | $50$               |
