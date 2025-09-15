---
hackmd:
  url: https://hackmd.io/0LDdYuOgQOylcjxiHSdoOg
  title: 2048 TD(0) HW1
  lastSync: 2025-09-15T04:37:08.630Z
---
# 2048 TD(0) HW1

## Report (20% + Bonus 20%)

-  A plot shows scores (mean) of at least 100k training episodes (20%)

![mean_score_log_1e6](https://hackmd.io/_uploads/ryeWuZHige.png)


## Bonus: (20%)

### Describe the implementation and the usage of 𝑛-tuple network. (5%)

First, `Pattern` records the target indices and compute the isomorphic patterns. It is a feature which means there is a set of weights underlying and we can train these weights as a n-tuple network. There are two operations to utilize the network, estimation and update. We can use `indexof` function and `operator[](index)` to get the target weight. For estimation, considered isomorphism, we see through each isomorphic patterns and sum them up. For update, we estimate the value based on the old weights, update the weights equally, and return the updated value.

For estimation,
```cpp
// use indexof the get the index of the isomorphic patterns
// sum up the weights.
size_t index = indexof(isomorphic[i], b);
weight_sum += operator[](index);
```

For update,
```cpp
// update the weights per isomorphic patterns.
size_t index = indexof(isomorphic[i], b);
operator[](index) += per;
```

### Explain the mechanism of TD(0). (5%)

Different from MC, it update the state value function during the each episode with its neighbor state. We update  $V(S_t)$ toward TD target $R_{t+1}+ \gamma V(S_{t+1})$. The formula with learning rate $\alpha$ is 

$$
V(S_t) \leftarrow V(S_t) + \alpha (R_{t+1} + \gamma V(S_{t+1}) - V(S_t))
$$

In this work, we adopt beforestate mechanism as the following picture.
We update beforestate value function with the next beforestate that the policy acted and the environment poped up a tile.  

![image](https://hackmd.io/_uploads/r1BqJGHjgx.png)

Compared with afterstate, which update the current afterstate value function with next afterstate value.

![image](https://hackmd.io/_uploads/HyY2kGHjgl.png)

When it comes to making a decision, expectation value is applied to find the action that produce the max expectation value:

$$
\pi(s) = argmax_a(R_{t+1} + \mathbb{E}[V(S_{t+1} \mid S_t=s, A_t=a)]) 
$$

### Describe your implementation in detail including action selection and TD-backup diagram. (10%) 

#### action selection

The probability of pop up 2 tile is 80%. 4 tile is 20%. With the formula in mind:

$$
\pi(s) = argmax_a(R_{t+1} + \mathbb{E}[V(S_{t+1} \mid S_t=s, A_t=a)]) 
$$

we iterate each action to simulate the environment action with the pop up probability to find the best action. The simulation can be done with the afterstate and the positions of empty tiles. It is easy to calculate the expection value once the next beforestate is simulated. Just estimate the feature on the next beforestate.

```cpp
// get the afterstate after action.
board s_prime = move->after_state();
```

```cpp
// find the empty tiles
std::vector<int> empty;
for(int i = 0; i < 16; i++){
	if(s_prime.at(i) == 0)
		empty.push_back(i);
}
```

```cpp
// simulate the popup
board next_popup2 = s_prime;
board next_popup4 = s_prime;
next_popup2.set(j, 1);
next_popup4.set(j, 2);
// calculate expection based on the simulation
EV += (0.8f*estimate(next_popup2) + 0.2f*estimate(next_popup4));
```

## TD-backup

![image](https://hackmd.io/_uploads/r1BqJGHjgx.png)

As previous description, we utilize the beforestate update. Updating from the end of the episode to the beginning one,  we apply the formula.

$$
V(S_t) \leftarrow V(S_t) + \alpha (R_{t+1} + \gamma V(S_{t+1}) - V(S_t))
$$

Shown as the code.

```cpp
float reward = rit->reward() != -1 ? rit->reward() : 0.0f;
float grad = reward - estimate(rit->before_state());
auto next_s = rit-1;
if (rit != path.rbegin()){
	grad += estimate(next_s->before_state());
}
update(rit->before_state(), alpha * grad);
```
