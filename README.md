# Genetic ALG Labs

The goal of today's labs is to use a genetic alg for generating a move patten of a sailor over a see.

You are given an environment, grid of rewards, that looks like a sea, the more blue a tile is the lower (more negative)
reward at this tile is.

Your job, is to generate a grid of arrow, than when an agent lands on a given spot, he knows where to move.

Your agent is always starting from a random of a left most field, and trying to go right, to fields that look yellow ish
on the map.

## Environment

The Environment is not deterministic, meaning random. 

On a given tile, with a given action you are not guarantee to perform said action.
When agent is on a tile T, with action A, there is 1% chance that the action will be inverted
(action up, will be down), there is 30% chance your action will be stared sideways, left or right (if action is up, you
will go left, ...). NOTE. This randomness is calculated each time you take an action! On tile T, you can be pushed back
the first time you visit it, but pushed left the second time.
Over all the chance that learned action is taken is 69% nice.

You also are penalize for hitting walls of the env.

_Note: Randomness is done using numpy_

## What has to be done?

You are to as per usual tweak the hyperparameters of the alg:
1. Number of agents in iteration (epoch)
2. Number of episodes (how many times agents see the env)
3. Probability of crossover happening
4. Probability of mutation happening
5. Whether the best individual proceeds with no change
6. How much better individuals should be more present in next generation (confusing)
7. Discounting parameter of rewards

To implement in the code, file `sailor_ag.py`:
- individual weighs with fitness values and parameter 5
- Cross over, parameter 3
- Mutation, parameter 4
- Parameter 5, and some additional stuff

After that there is only hyperparameter tuning left.
