# DQN-TicTacToe
my demo of DQN with TicTacToe

### The Agent 
The agent in the demo is special in the sense that it doesn't only do the right moves for only one player (X\O) but predict the best move in each position **whatever the turn**.

In each position `s` the agent will predict the best action `a` using the Q-learning equation: 
* a = argmax_a'(Q(s, a'))

### The Training
In order to train the agent we will use a relay memory approach which will sample random Transitions (current state, action, reward, next state. where current state isn't terminal) and train the agent on them.
for a given Transition the agent be trained using the following equation:
if next state isn't terminal:
* Q(current_state, action) = reward - max_a'(Q(next_state, a'))

Translating that to human language we get "the best that i can do in the current position is the reward to the next position plus the worst action (to me) that my enemy can do to me". in our case we can remove the reward term since any reward to a non-terminal state is always 0, so will use:
* Q(current_state, action) = - max_a'(Q(next_state, a'))

if next state is terminal:
* Q(current_state, action) = reward