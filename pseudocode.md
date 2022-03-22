# Project Pseudocode

## Training loop
```
num_skills := 20
# Number of sample trajectories to collect at the end of each skills
num_collected_trajectories := 100


def transform_observation(observation, environment):
	# Return the global state encoding which determines the intrinsic reward


# The buffers are initialized with the shapes
replay_buffer = ReplayBuffer(observations, transformed_obs, action, next_observations, next_transformed_obs, timestep)
reward_buffer = RewardBuffer(transformed_obs)


# Training loop
for i in 1...num_skills:
	initialize agent[i] := (actor[i], critic[i])
	
	# Initialize reward buffer for a new skill
	reward_buffer.init_skill()
	
	for ep in 1...total_episodes:  #  In practice we run for total number of steps, not episodes
		observation = environment.reset()
		transformed_obs = transform_observation(observation, environment)
		timestep = 0
		while not done:
			# Step
			action = actor[i].act(observation)
			next_observation, _reward, done  = environment.step(action)
			next_transformed_obs = transform_observation(observation, environment)

			# Update buffers
			replay_buffer.add(observation, transformed_obs, action, next_observation, next_transformed_obs, timestep)
			reward_buffer.add_current(transformed_obs, next_transformed_obs, timestep)

			agent[i].update()

			timestep += 1
			observation, transformed_obs = next_observation, next_transformed_obs
			
	# Uncomment this line to reset the replay buffer at each skill
	# replay_buffer = ReplayBuffer(observations, transformed_obs,
					  			   action, next_observations,
								   next_transformed_obs, timestep)
	
	# Collect trajectories for the reward buffer
	for ep in 1...num_collected_trajectories:
		observation = environment.reset()
		transformed_obs = transform_observation(observation, environment)
		timestep = 0
		while not done:
			# Step
			action = actor[i].act(observation)
			next_observation, _reward, done = environment.step(action)
			next_transformed_obs = transform_observation(observation, environment)
      
			# Update buffers
			reward_buffer.add_old(transformed_obs, next_transformed_obs, timestep)
      
			agent[i].update()
			timestep += 1
			observation, transformed_obs = next_observation, next_transformed_obs
```

---

## Agent updating:

```
def update():
	(observations, transformed_obs,
	 action, next_observations,
	 next_transformed_obs, timestep) = replay_buffer.sample()
	 
	 reward = reward_buffer.compute_reward(transformed_obs, next_transformed_obs, timestep)
	 
	 update_actor(observations, action, next_observations, reward)
	 update_critic(observations, action, next_observations, reward)
```

---

## Reward module
This part of the code uses some tricky matrix manipulation to vectorize the following operations, but I am taking them apart for ease of understanding.

```
# we only consider this many trajectories to compute the consistency reward
max_current_trajectories := 100

def init_skill():
	current_trajectory_buffer.clear()
	
def add_current(...):
	current_trajectory_buffer[timestep].add(transformed_obs, next_transformed_obs)

def add_old(...):
	old_trajectory_buffer[timestep].add(transformed_obs, next_transformed_obs)
	
def compute_reward(transformed_obs, next_transformed_obs, timestep):	
	# Consistency penalty
	current_skill_obses, current_skill_next_obses = current_trajectory_buffer[timestep].sample()
	consistency_penalty = compute_entropy(next_transformed_obs, current_skill_next_obses)
	
	# Diversity reward
	past_skill_obses, past_skill_next_obses = old_trajectory_buffer[timestep].sample()
	diversity_reward = compute_entropy(next_transformed_obs, past_skill_next_obses)
	
	return (beta * diversity_reward - alpha * consistency_penalty)
	
	
def compute_entropy(obs_vector, set_of_obs, k=5):
	# Estimate the pointwise entropy of a set
	# Return the k-th top euclidean distance
	return topk(obs_vector - set_of_obs, k=k)[-1]

```
