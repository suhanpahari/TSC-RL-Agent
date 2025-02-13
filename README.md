
# Deep Reinforcement Learning Policy Comparison

<p align="center"> <img src="https://www.gymlibrary.dev/_images/pong.gif" alt="Pong" /> </p>



## Overview

This experimental project implements and compares four different action selection policies for Deep Q-Networks (DQN) in the "PongNoFrameskip-v4" environment using OpenAI Gym. The four policies evaluated are:

1. **Epsilon-Greedy**: Selects the action with the highest Q-value with probability $1 - \epsilon$ and a random action otherwise.

   <!-- $$
   a = \begin{cases}
   \arg\max_a Q(s, a), & \text{with probability } (1 - \epsilon) \\
   \text{random action}, & \text{with probability } \epsilon
   \end{cases}
   $$ -->

2. **Boltzmann Exploration**: Uses a softmax function over Q-values to determine action probabilities.

   <!-- $$
   P(a) = \frac{e^{Q(s,a)/T}}{\sum_{b} e^{Q(s,b)/T}}
   $$ -->

   where $T$ is the temperature parameter controlling exploration.

3. **Upper Confidence Bound (UCB)**: Adjusts Q-values using an exploration bonus based on action visit counts.

   <!-- $$
   Q_{UCB}(s, a) = Q(s, a) + c \sqrt{\frac{\ln(t)}{N(s, a)}}
   $$ -->

   where $c$ is a confidence parameter, $t$ is the total number of steps, and $N(s, a)$ is the count of action $a$ taken in state $s$.

4. **Thompson Sampling**: Samples Q-values from a normal distribution and selects the highest.

   <!-- $$
   Q_{TS}(s, a) \sim \mathcal{N}(Q(s, a), \sigma^2(s, a))
   $$ -->

   where $\sigma^2(s, a)$ is the variance estimate of the Q-value.
    

## Implementation Details

-   The project utilizes **PyTorch** for deep learning model implementation.
-   The **DQN architecture** consists of convolutional layers followed by fully connected layers.
-   **Frame preprocessing** includes grayscale conversion and resizing to 84x84.
-   **Replay memory** stores experience tuples and enables batch training.
-   **Parallel training** is implemented using `multiprocessing` for efficiency.

## Code Structure

-   `DQN(nn.Module)`: Defines the convolutional neural network for Q-learning.
-   `pre(f)`: Preprocesses frames for input.
-   `Stack`: Maintains a stack of previous frames for temporal information.
-   `Mem`: Implements experience replay memory.
-   `train()`: Trains a DQN model using the selected policy.
-   `main()`: Runs multiple training processes with different policies and collects results.

## Dependencies

Ensure the following dependencies are installed:

```bash
pip install gym torch numpy opencv-python multiprocessing

```

## Usage

Run the script using:

```bash
python main.py

```

The program will train separate DQN models using each policy and output performance metrics, including total rewards and exploration parameters over time.

## Results

The final output will compare the performance of each policy by displaying:

-   Total rewards per episode.
-   The evolution of exploration parameters (epsilon, temperature, etc.).
-   A side-by-side comparison of policy effectiveness.

## Future Work

-   Implement additional policies like Soft Q-learning or Bayesian DQN.
-   Tune hyperparameters for better convergence.
-   Test in more complex environments beyond Pong.

## License

This project is released under the MIT License.