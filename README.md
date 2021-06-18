# RL for Goal Recognition

This is a projet intended to apply reinforcement learning method to goal recognition problems.

## Methods

### Tabular Q-Learning

Useful for environments with a small state-space (10K tops), presents good learning results and, therefore, a good accuracy on goal recognition. This was tested on the Blocksworld environment using up do 4 blocks, and for 7 blocks or more presented poor policy learning, pretty much never reaching the objective goal.

## TODO Notes

Research:

- use epsilon-greedy policies instead of 1 for maximum, 0 otherwise
- test with negative rewards
- policy gradient methods

Code and functionality:

- Serialization (State and actions to their string representation, map them back with a dictionary)