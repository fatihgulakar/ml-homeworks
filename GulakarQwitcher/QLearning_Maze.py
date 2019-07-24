
import sys
import random
import time
import TinyMaze
import matplotlib.pyplot as plt
from scipy import ndimage


# rewards = 1: "stepped": -1; 2: "blocked": -2; 3: "won": +100; 5: "poisoned": -100
rewards = {1: -1, 2: -5, 3: 100, 5: -100}

# actions = 0: "Left": "a"; 1: "Right": "s"; 2: "Up": "w"; 3: "Down": "z"
actions = {0:'a', 1:'s', 2:'w', 3:'z'}

# define maze size (NxN)
maze_size = 11

# define training exploration chance
exploration_level = 0.2

# set up Q table: NXN rows (all positions), 4 cols (all moves)
Q = [ [0 for a in actions] for s in range(maze_size * maze_size) ]

# define learning rate and discount rate
learning_rate = 0.7	
discount_rate = 0.9

# define the number of steps and episodes		
num_steps = 100
num_epsiodes = 100
episode_rewards = []



# explore maze during each training episode
for i in range(num_epsiodes):

	# reset the maze and total reward
	env = TinyMaze.TinyMazeEnv(maze_size)
	total_reward = 0

	# s = env.maze_size * env.y + env.x
	# max_reward = max(Q[s])
	# if random.uniform(0,1) > exploration_level:
	#     action_indexS = Q[s].index(max_reward)
	# else:
	# 	action_indexS = random.randint(0,len(actions)-1)

	# step through maze
	for j in range(num_steps):

		# calculate current state based on x, y position
		s = env.maze_size * env.y + env.x

		# get maximum total reward for this state using current Q table
		max_reward = max(Q[s])

		# either explore (random action) or exploit (highest reward)
		if random.uniform(0,1) > exploration_level:
			# select action index based on maximum total reward in Q table
			action_index = Q[s].index(max_reward)
		else:
			action_index = random.randint(0,len(actions)-1)

		# make action and get specific reward
		status = env.step(actions[action_index])
		action_reward = rewards[status]
		total_reward += action_reward

		# apply Bellman equation
		new_state = env.maze_size * env.y + env.x
		# Update Q table using Bellman equation
		Q[s][action_index] += learning_rate*(action_reward + discount_rate*max(Q[new_state]) - Q[s][action_index])
		
		#Q[s][action_index] += learning_rate*(action_reward + discount_rate*Q[new_state, action_indexS] - Q[s][action_index])
		# if we won the game, then stop the episode
		if status == env.won:
			break

	# track total reward vs episode
	episode_rewards.append(total_reward)



# demonstrate trained solution
env = TinyMaze.TinyMazeEnv(maze_size)
count = 0
while True:
	s = env.maze_size * env.y + env.x
	max_reward = max(Q[s])
	action_index = Q[s].index(max_reward)
	status = env.step(actions[action_index])
	env.display_maze(actions[action_index])
	if status == env.won:
		break
	count += 1
	if count > 100:
		break
	time.sleep(0.25)

# display training plot and Q table

plt.figure()
plt.imshow(env.mazes[maze_size],cmap='Greys')
for i in range(maze_size):
	for j in range(maze_size):
		text = plt.text(j,i,i*maze_size+j,ha='center',va='center',color='w',fontsize=8)
text = plt.text(maze_size-1,maze_size-1,"END",ha='center',va='center',color='k',fontsize=8)
plt.xticks([])
plt.yticks([])

# plot results of training
plt.figure()
plt.plot(episode_rewards,'-')
plt.xlabel('Episode #')
plt.ylabel('Reward')

# plot Q table
plt.figure()
img  = plt.imshow(Q)
plt.colorbar(img)
plt.title("Q table")
plt.imshow(ndimage.rotate(Q, -90),aspect=4.5)
plt.yticks([0,1,2,3])
plt.ylabel("0:Left 1:Right 2:Up 3:Down")


plt.show()
