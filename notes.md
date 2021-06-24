# Self notes (or development updates and reminders)

## Architectures tested:

### Arch. 1:
input -> 128 -> 256 -> 128 -> actions
did not reach goal after a long time
Learned to not use many invalid actions
top reward: -50 (no invalid actions performed)
Loss got really small

Hyperparameters:
- discount: 0.99
- annealing epsilon until: 200000
- end epsilon: 0.1
- timesteps: 50
- steps before training: 5

### Arch. 2:
input -> 128 -> 128 -> 128 -> actions
simpler model for simpler problems.
same thing as arch. 1, learned to not perform invalid actions
but did not find goal
most likely the reason would be that there is no indication to where the goal would be
in a future experiment, test some kind of reward function that stimulates the agent to perform
actions that lead to the goal being reached.

Hyperparameters:
- discount: 0.99
- annealing epsilon until: 200000
- end epsilon: 0.1
- timesteps: 50
- steps before training: 5

### Arch. 3

A recommendation by Meneguzzi, but don't really think this will lead to anything.
Same neural net architecture, but change the reward function to the original PDDLGym one:
+1 if reached goal, 0 otherwise.
Test multiple discount factors.

#### gamma = 0.99

As expected, nothing happens. Will start to track how many different states were found, just to understand what is happening with the search space.

Hyperparameters:
- discount: 0.99
- annealing epsilon until: 200000
- end epsilon: 0.1
- timesteps: 50
- steps before training: 5

### Arch. 4 - Positive rewards for partially reached goals

Each time the agent finds an action that partially reaches its goal, a positive reward is earned.

With basic model, loss explodes.

Will develop some improved DQN architectures (dueling, double)

After adding dueling DQN:

loss is still completely fucked