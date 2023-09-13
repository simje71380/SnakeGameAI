import torch
import random
import numpy as np
from collections import deque
import SnakeGame
from model import Linear_QNet, Linear_QNet2, QTrainer
from helper import plot
import keyboard, os

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001
EXPLORATION_RATE_BEGIN = 80 #80% exploration
EXPLORATION_DECREASE_RATE = 0.6

GET_STATE_GRID = 0
GET_STATE_DANGER_SEE_1 = 1
GET_STATE_DANGER_SEE_2 = 2
GET_STATE_DANGER_3_AROUND = 3

class Agent:

    def __init__(self, mode=GET_STATE_DANGER_SEE_1):
        self.n_games = 0
        self.epsilon = 0 # randomness
        self.gamma = 0.9 # discount rate
        self.memory = deque(maxlen=MAX_MEMORY) # popleft()
        if(mode == GET_STATE_GRID):
            self.model = Linear_QNet(SnakeGame.CELL_NUMBER*SnakeGame.CELL_NUMBER + 11, 256, 3)
        elif(mode == GET_STATE_DANGER_SEE_1):
            self.model = Linear_QNet(11, 32, 3)
        elif(mode == GET_STATE_DANGER_SEE_2):
            self.model = Linear_QNet(11 + 7, 128, 3)
        elif(mode == GET_STATE_DANGER_3_AROUND):
            self.model = Linear_QNet(56, 256, 3)
        else:
            print("unknown mode")
        self.mode = mode
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)

    def load_model(self):
        model_folder_path = './model'
        if not os.path.exists(model_folder_path):
            print("Error : unable to locate model directory")
            return



        if(self.mode == GET_STATE_GRID):
            file_name = f"model_grid_{SnakeGame.CELL_NUMBER}.pth"
        elif(self.mode == GET_STATE_DANGER_SEE_1):
            file_name = f"model_danger_1_{SnakeGame.CELL_NUMBER}.pth"
        elif(self.mode == GET_STATE_DANGER_SEE_2):
            file_name = f"model_danger_2_{SnakeGame.CELL_NUMBER}.pth"
        elif(self.mode == GET_STATE_DANGER_3_AROUND):
            file_name = f"model_danger_3_around_{SnakeGame.CELL_NUMBER}.pth"
        else:
            print("unknown mode")

        file_name = os.path.join(model_folder_path, file_name)
        self.model.load_state_dict(torch.load(file_name))

    def get_state(self, game):
        if(self.mode == GET_STATE_GRID):
            return self._get_state_grid(game)
        elif(self.mode == GET_STATE_DANGER_SEE_1):
            return self._get_state_danger_see_1(game)
        elif(self.mode == GET_STATE_DANGER_SEE_2):
            return self._get_state_danger_see_2(game)
        elif(self.mode == GET_STATE_DANGER_3_AROUND):
            return self._get_state_danger_3_around_head(game)
        else:
            print("unknown mode")
        

    def _get_state_grid(self, game):
        #state -> danger : straight, left, right   moving direction   food position   grid reshape(CELL_NUMBER^2, 1)

        grid = game.grid
        head = game.snake[0]
        try:
            head_up = grid[head[0], head[1] - 1] == 1 or head[1] - 1 == -1
        except:
            head_up = True
        try:
            head_down = grid[head[0], head[1] + 1] == 1 or head[1] + 1 == 20
        except:
            head_down = True
        try:
            head_left = grid[head[0] - 1, head[1]] == 1 or head[0] - 1 == -1
        except:
            head_left = True
        try:
            head_right = grid[head[0] + 1, head[1]] == 1 or head[0] + 1 == 20
        except:
            head_right = True

        
        
        dir_l = game.direction == SnakeGame.DIRECTIONS_CLOCKWISE[3] #left
        dir_r = game.direction == SnakeGame.DIRECTIONS_CLOCKWISE[1] #right
        dir_u = game.direction == SnakeGame.DIRECTIONS_CLOCKWISE[0] #up
        dir_d = game.direction == SnakeGame.DIRECTIONS_CLOCKWISE[2] #down


        state = [
            # Danger straight
            (dir_r and head_right) or 
            (dir_l and head_left) or 
            (dir_u and head_up) or 
            (dir_d and head_down),

            # Danger right
            (dir_u and head_right) or 
            (dir_d and head_left) or 
            (dir_l and head_up) or 
            (dir_r and head_down),

            # Danger left
            (dir_d and head_right) or 
            (dir_u and head_left) or 
            (dir_r and head_up) or 
            (dir_l and head_down),
            
            # Move direction
            dir_l,
            dir_r,
            dir_u,
            dir_d,
            
            # Food location 
            game.food[0] < head[0],  # food left
            game.food[0] > head[0],  # food right
            game.food[1] < head[1],  # food up
            game.food[1] > head[1]  # food down
            ]

        grid = game.grid.reshape(SnakeGame.CELL_NUMBER*SnakeGame.CELL_NUMBER,1)
        for x in np.nditer(grid):
            state.append(x)

        return np.array(state, dtype=int)
    
    def _get_state_danger_3_around_head(self, game):
        head = game.snake[0]
        grid = game.grid

        '''
        0000000
        0000000
        0000000
        0001000  1 = head | 0 = free space
        0000000
        0000000
        0000000
        '''
        view = np.zeros((7,7), dtype=int)

        for i in range(-3,4):
            for j in range(-3,4):
                try:
                    if grid[head[0]+i, head[1] + j] == SnakeGame.BODY:
                        view[i+3, j+3] = 1
                except:
                    view[i+3, j+3] = 1

        view = np.reshape(view, view.shape[0]*view.shape[1])
        view = np.delete(view, 24)

        dir_l = game.direction == SnakeGame.DIRECTIONS_CLOCKWISE[3] #left
        dir_r = game.direction == SnakeGame.DIRECTIONS_CLOCKWISE[1] #right
        dir_u = game.direction == SnakeGame.DIRECTIONS_CLOCKWISE[0] #up
        dir_d = game.direction == SnakeGame.DIRECTIONS_CLOCKWISE[2] #down

        state = [        
            # Move direction
            dir_l,
            dir_r,
            dir_u,
            dir_d,
            
            # Food location 
            game.food[0] < head[0],  # food left
            game.food[0] > head[0],  # food right
            game.food[1] < head[1],  # food up
            game.food[1] > head[1]  # food down
            ]

        for x in np.nditer(view):
            state.append(x)

        return np.array(state, dtype=int)

    def _get_state_danger_see_1(self, game):
        head = game.snake[0]
        grid = game.grid

        try:
            head_up = grid[head[0], head[1] - 1] == SnakeGame.BODY or head[1] - 1 == -1
        except:
            head_up = True
        try:
            head_down = grid[head[0], head[1] + 1] == SnakeGame.BODY or head[1] + 1 == 20
        except:
            head_down = True
        try:
            head_left = grid[head[0] - 1, head[1]] == SnakeGame.BODY or head[0] - 1 == -1
        except:
            head_left = True
        try:
            head_right = grid[head[0] + 1, head[1]] == SnakeGame.BODY or head[0] + 1 == 20
        except:
            head_right = True

        
        
        dir_l = game.direction == SnakeGame.DIRECTIONS_CLOCKWISE[3] #left
        dir_r = game.direction == SnakeGame.DIRECTIONS_CLOCKWISE[1] #right
        dir_u = game.direction == SnakeGame.DIRECTIONS_CLOCKWISE[0] #up
        dir_d = game.direction == SnakeGame.DIRECTIONS_CLOCKWISE[2] #down

        state = [
            # Danger straight
            (dir_r and head_right) or 
            (dir_l and head_left) or 
            (dir_u and head_up) or 
            (dir_d and head_down),

            # Danger right
            (dir_u and head_right) or 
            (dir_d and head_left) or 
            (dir_l and head_up) or 
            (dir_r and head_down),

            # Danger left
            (dir_d and head_right) or 
            (dir_u and head_left) or 
            (dir_r and head_up) or 
            (dir_l and head_down),
            
            # Move direction
            dir_l,
            dir_r,
            dir_u,
            dir_d,
            
            # Food location 
            game.food[0] < head[0],  # food left
            game.food[0] > head[0],  # food right
            game.food[1] < head[1],  # food up
            game.food[1] > head[1]  # food down
            ]
        return np.array(state, dtype=int)
    

    def _get_state_danger_see_2(self, game):
        head = game.snake[0]
        grid = game.grid


        #see 1 case ahead
        try:
            head_up = grid[head[0], head[1] - 1] == SnakeGame.BODY or head[1] - 1 == -1
        except:
            head_up = True
        try:
            head_down = grid[head[0], head[1] + 1] == SnakeGame.BODY or head[1] + 1 == 20
        except:
            head_down = True
        try:
            head_left = grid[head[0] - 1, head[1]] == SnakeGame.BODY or head[0] - 1 == -1
        except:
            head_left = True
        try:
            head_right = grid[head[0] + 1, head[1]] == SnakeGame.BODY or head[0] + 1 == 20
        except:
            head_right = True

        #see 2 cases ahead
        try:
            head_up20 = grid[head[0]-1, head[1] - 2] == SnakeGame.BODY or head[1] - 2 == -1
        except:
            head_up20 = True
        try:
            head_up21 = grid[head[0], head[1] - 2] == SnakeGame.BODY or head[1] - 2 == -1
        except:
            head_up21 = True
        try:
            head_up22 = grid[head[0]+1, head[1] - 2] == SnakeGame.BODY or head[1] - 2 == -1
        except:
            head_up22 = True

                
        try:
            head_down20 = grid[head[0]-1, head[1] + 2] == SnakeGame.BODY or head[1] + 2 == 20
        except:
            head_down20 = True
        try:
            head_down21 = grid[head[0], head[1] + 2] == SnakeGame.BODY or head[1] + 2 == 20
        except:
            head_down21 = True
        try:
            head_down22 = grid[head[0]+1, head[1] + 2] == SnakeGame.BODY or head[1] + 2 == 20
        except:
            head_down22 = True

        try:
            head_left2 = grid[head[0] - 2, head[1]] == SnakeGame.BODY or head[0] - 2 == -1
        except:
            head_left2 = True
        try:
            head_right2 = grid[head[0] + 2, head[1]] == SnakeGame.BODY or head[0] + 2 == 20
        except:
            head_right2 = True

        
        
        dir_l = game.direction == SnakeGame.DIRECTIONS_CLOCKWISE[3] #left
        dir_r = game.direction == SnakeGame.DIRECTIONS_CLOCKWISE[1] #right
        dir_u = game.direction == SnakeGame.DIRECTIONS_CLOCKWISE[0] #up
        dir_d = game.direction == SnakeGame.DIRECTIONS_CLOCKWISE[2] #down

        state = [
            # Danger straight
            (dir_r and head_right) or 
            (dir_l and head_left) or 
            (dir_u and head_up) or 
            (dir_d and head_down),

            # Danger right
            (dir_u and head_right) or 
            (dir_d and head_left) or 
            (dir_l and head_up) or 
            (dir_r and head_down),

            # Danger left
            (dir_d and head_right) or 
            (dir_u and head_left) or 
            (dir_r and head_up) or 
            (dir_l and head_down),

            # Danger 2 straight
            (dir_r and head_right2) or 
            (dir_l and head_left2) or 
            (dir_u and head_up21) or 
            (dir_d and head_down21),
            
            # Danger 2 right
            (dir_u and head_right2) or 
            (dir_d and head_left2) or 
            (dir_l and head_up21) or 
            (dir_r and head_down21),

            # Danger 2 left
            (dir_d and head_right2) or 
            (dir_u and head_left2) or 
            (dir_r and head_up21) or 
            (dir_l and head_down21),

            # Danger 2 straight left
            (dir_r and head_up22) or 
            (dir_l and head_down20) or 
            (dir_u and head_up20) or 
            (dir_d and head_down22),
            
            # Danger 2 straight right
            (dir_u and head_up22) or 
            (dir_d and head_down20) or 
            (dir_l and head_up20) or 
            (dir_r and head_down22),

            # Danger 2 down left
            (dir_r and head_up20) or 
            (dir_l and head_down22) or 
            (dir_u and head_down20) or 
            (dir_d and head_up22),
            
            # Danger 2 down right
            (dir_u and head_down22) or 
            (dir_d and head_up20) or 
            (dir_l and head_up22) or 
            (dir_r and head_down20),

            
            # Move direction
            dir_l,
            dir_r,
            dir_u,
            dir_d,
            
            # Food location 
            game.food[0] < head[0],  # food left
            game.food[0] > head[0],  # food right
            game.food[1] < head[1],  # food up
            game.food[1] > head[1]  # food down
            ]

        return np.array(state, dtype=int)



    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done)) # popleft if MAX_MEMORY is reached

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE) # list of tuples
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)
        #for state, action, reward, nexrt_state, done in mini_sample:
        #    self.trainer.train_step(state, action, reward, next_state, done)

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state):
        # random moves: tradeoff exploration / exploitation
        self.epsilon = EXPLORATION_RATE_BEGIN - EXPLORATION_DECREASE_RATE*self.n_games
        final_move = [0,0,0]
        if random.randint(0, 100) < self.epsilon:
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1

        return final_move

    def get_action_NN(self, state):
        final_move = [0,0,0]
        state0 = torch.tensor(state, dtype=torch.float)
        prediction = self.model(state0)
        move = torch.argmax(prediction).item()
        final_move[move] = 1
        return final_move

def train(mode):
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    agent = Agent(mode)
    agent.load_model()
    game = SnakeGame.SnakeGame(SnakeGame.INPUT_AI)
    stop = False
    while not stop:
        # get old state
        state_old = agent.get_state(game)

        # get move
        final_move = agent.get_action(state_old)

        # perform move and get new state
        running, score, reward = game.playStep(final_move)
        done = not running
        #print(f"done : {done}  score : {score}  reward : {reward}")
        state_new = agent.get_state(game)

        # train short memory
        agent.train_short_memory(state_old, final_move, reward, state_new, done)

        # remember
        agent.remember(state_old, final_move, reward, state_new, done)

        if done:
            # train long memory, plot result
            if(final_move == [1, 0 ,0]):
                move = "straight"
            if(final_move == [0, 1 ,0]):
                move = "left"
            if(final_move == [0, 0 ,1]):
                move = "right"
            #print(f"tried to go : {move} while state is : {state_old}")
            game = SnakeGame.SnakeGame(SnakeGame.INPUT_AI) #new game
            agent.n_games += 1
            agent.train_long_memory()

            if score > record:
                record = score
                if(agent.mode == GET_STATE_GRID):
                    file_name = f"model_grid_{SnakeGame.CELL_NUMBER}.pth"
                elif(agent.mode == GET_STATE_DANGER_SEE_1):
                    file_name = f"model_danger_1_{SnakeGame.CELL_NUMBER}.pth"
                elif(agent.mode == GET_STATE_DANGER_SEE_2):
                    file_name = f"model_danger_2_{SnakeGame.CELL_NUMBER}.pth"
                elif(agent.mode == GET_STATE_DANGER_3_AROUND):
                    file_name = f"model_danger_3_around_{SnakeGame.CELL_NUMBER}.pth"
                else:
                    print("unknown mode")
                agent.model.save(file_name)

            print('Game', agent.n_games, 'Score', score, 'Record:', record)

            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.n_games
            plot_mean_scores.append(mean_score)
            plot(plot_scores, plot_mean_scores)
        
        if keyboard.is_pressed("q"):
            stop = True
            print("saving best model")