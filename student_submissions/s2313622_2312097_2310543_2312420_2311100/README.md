- To run the trained RL PPO policy: set `policy_id` to 1
- To run the hybrid policy (heuristic + RL): set `policy_id` to 2
- To run the heuristic policy: set `policy_id` to 3


## Instructions for the RL PPO policy (policy_id=1)

**ACKNOWLEDGEMENT**: This code is based on the PPO implementation from the following source: https://github.com/ericyangyu/PPO-for-Beginners

The file `ppo_optimized.py` contains the code necessary for training and running the policy for the Reinforcement Learning algorithm: in this case, Proximal Policy Optimization (PPO). The policy network's parameters are stored in `ppo_actor.pth` and `ppo_critic.pth`.

First, you need to install the required Python libraries through:

```
pip install -r requirements.txt
```

The current `ppo_actor.pth` and `ppo_critic.pth` files contain the parameters of the trained PPO policy for the `gym_cutting_stock` environment provided in this assignment, and it has already been trained on our computer in advanced. We have trained the network for the default `gym_cutting_stock` configuration (min_w=50, min_h=50, max_w=100, max_h=100, num_stocks=100, max_product_type=25), so if you are testing this RL policy with the default configuration, you don't have to retrain the model.

However, if you want to use this RL policy for a different `gym_cutting_stock` environment configuration (such as changing num_stocks to 50, changing max_w to 80,...), you will need to retrain the PPO model, because the deep learning network implemented for the policy takes in fixed input and output sizes. Depending on the size of your new environment and your computer's technical specifications, the training process may need between 30 minutes to multiple hours. Also, this deep learning network used for the policy may use more than 8GB of RAM during training, so you might have to lower the values `timesteps_per_batch` and `max_timesteps_per_episode` at line 802 and 803 of `ppo_optimized.py` if your computer can't keep up.

If you want to train this RL policy, first navigate to this directory in your terminal:

```
cd student_submissions/s2313622_2312097_2310543_2312420_2311100/
```

**NOTE**: While training, `ppo_optimized` will overwrite the existing files `ppo_actor.pth` and `ppo_critic.pth` in this folder.

- To train from scratch:
```
python ppo_optimized.py
```


- To train with existing actor/critic models:
```
python ppo_optimized.py --actor_model ppo_actor.pth --critic_model ppo_critic.pth
```

- To run the policy for testing: Go back to the original `main.py` file in the parent directory (run `cd ../..` then simply `python main.py`) and call the policy with policy_id = 1. Remember to use the correct configuration of `gym_cutting_stock` environment while running.

## Instructions for the Heuristic_RL Policy

The policy does not require prior training. During testing, it trains on a single specific episode provided by the user and then proceeds to testing. This on-the-fly training allows the policy to quickly adapt to the given environment. Training with the copied environment, optimized using the heuristic method, takes less than 10 seconds.

To train and test the policy, go back to the original main.py file in the parent directory and call the policy with policy_id = 2.

**NOTE**: The user must run an episode starting with info["filled_ratio"] == 0. This ensures that the policy can train properly and reset necessary information.

#### Instructions for the Heuristic Policy

To test the policy, go back to the original main.py file in the parent directory and call the policy with policy_id = 3.

**NOTE**: The user must run an episode starting with info["filled_ratio"] == 0. This ensures that the policy resets necessary information.