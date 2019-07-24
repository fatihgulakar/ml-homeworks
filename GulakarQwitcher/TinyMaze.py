import sys
import random

class TinyMazeEnv():

	# define status codes
	stepped = 1
	blocked = 2
	won = 3
	quit = 4
	# define mazes
	mazes =	{11: [	[ 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0],
			 		[ 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1],
			 		[ 0, 0, 2, 1, 0, 0, 0, 0, 0, 1, 0],
			 		[ 1, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1],
			 		[ 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1],
			 		[ 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 0],
			 		[ 1, 0, 0, 2, 1, 0, 0, 0, 2, 1, 0],
			 		[ 1, 1, 0, 0, 0, 0, 1, 1, 0, 1, 0],
			 		[ 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1],
			 		[ 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
			 		[ 0, 0, 1, 0, 0, 1, 1, 1, 0, 0, -1] ]
			}

	poisoned=5
	def __init__(self,maze_size=5):
		# initialize starting position and maze
		self.x = 0
		self.y = 0
		self.total_steps = 0
		if maze_size in self.mazes.keys():
			self.maze = self.mazes[maze_size]
		else:
			self.maze = self.mazes[5]
		self.maze_size = len(self.maze)

	def display_maze(self,move='None'):
		# display the maze in its current state
		print("\n" * 100) 	# clear screen
		print("")
		print("---" * (self.maze_size + 2))
		for i in range(self.maze_size):
			row = " | "
			for j in range(self.maze_size):
				if i == self.y and j == self.x: 
					row += " G "
				elif self.maze[i][j] == 1: 
					row += "###"
				elif self.maze[i][j] == -1: 
					row += " B "
				elif self.maze[i][j] == 2: 
					row += " P "
				else:
					row += "   "

			print(row + " | ")
			print("---" * (self.maze_size + 2))
		print("Move: {0}  Total steps: {1}".format(move,self.total_steps))
		print("x: {0}, y: {1}".format(self.x,self.y))

	def step(self,move):
		# process a single action
		self.total_steps += 1
		status = self.blocked
		if move == "a":
			if (self.x > 0) and (self.maze[self.y][self.x-1] != 1): 
				self.x -= 1
				status = self.stepped
		elif move == "s":
			if (self.x < self.maze_size-1) and (self.maze[self.y][self.x+1] != 1): 
				self.x += 1
				status = self.stepped
		elif move == "w":
			if (self.y > 0) and (self.maze[self.y-1][self.x] != 1): 
				self.y -= 1
				status = self.stepped
		elif move == "z":
			if (self.y < self.maze_size-1) and (self.maze[self.y+1][self.x] != 1): 
				self.y += 1
				status = self.stepped
		elif move == "Q":
			status = self.quit

		# check for a win
		if self.maze[self.y][self.x] == -1:
			status = self.won
        # check for a poison
		if self.maze[self.y][self.x] == 2:
			status = self.poisoned
        
		return status
