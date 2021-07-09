# RL for Goal Recognition

This project has the main objective of applying (Inverse) Reinforcement Learning to the problem of Goal Recognition. Instead of computing goals the way standard algorithms do, we try to learn multiple policies via Reinforcement Learning to learn how to reach each of the possible goals on the environment, and then compute the error of a trajectory and the policies. The solution to the problem is the goal of the policy that presents the least error between all of them.

## Running the code

Install the requirements located at `requirements.txt`, source `setup.sh`, and then you can run either a training or a trace extraction. To train an agent, run `src/train.py`, selecting the type of agent you want to train. To extract traces for debugging and later on running training for imitation learning algorithms, run `src/dataset/extract_dataset.py`. For now, only **blocksworld** environment works, as PDDLGym has a renderer function implemented for it, and is one of the easiest to integrate with the Goal Recognition dataset.
## Methods

### Tabular Q-Learning

Useful for environments with a small state-space (10K tops), presents good learning results and, therefore, a good accuracy on goal recognition. This was tested on the Blocksworld environment using up to 4 blocks, and for 7 blocks or more presented poor policy learning, pretty much bruteforcing until reaching the goal.

### Linear Function Approximation Q-Learning

Tested with two variants: simple model using the selected action as a one-hot encoded vector on top of the state, and multiple models, one for each action. Second approach performed better, though presenting a large overhead when computing the q-value for all actions. Did not learn well.

### Deep Q-Learning

Good for large state-space environments. In this case though, learning did not succeed too well when considering invalid actions. This is due to neural networks not generalising properly when they see pretty much all states having reward 0, leading to a generalisation of always 0 to any state. Pretty much brute force/blind search through the environment.

## TODO Notes

Research:

- policy gradient methods
- Imitation learning (or Inverse Reinforcement Learning)
  - Behavioral cloning
  - 

Code and functionality:

- Serialization (State and actions to their string representation, map them back with a dictionary)