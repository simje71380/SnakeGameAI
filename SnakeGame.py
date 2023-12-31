import pygame
from pygame import *
import numpy as np
from random import *
import math as m

import vidmaker


#Window properties
WIN_WIDTH = 960
WIN_HEIGHT = 720
WIN_SCOREBOARD_OFFSET = WIN_WIDTH - (WIN_WIDTH - WIN_HEIGHT)

CELL_NUMBER = 20
CELL_SIZE = int(WIN_HEIGHT/CELL_NUMBER)

FLAGS = DOUBLEBUF


#Game properties
REFRESH_RATE = 2
REFRESH_RATE_AI = 30


HEAD_COLOR = (0, 128, 0)
BODY_COLOR = (0, 255, 0)
FOOD_COLOR = (255, 0, 0)
BG_COLOR = (0, 0, 0)
TEXT_COLOR = (255, 255, 255)
SNAKE_INIT_HEAD_POS = [int(CELL_NUMBER/2), int(CELL_NUMBER/2)]

HEAD = 1
BODY = 2
FOOD = 3

SNAKE_MAX_SIZE = CELL_NUMBER * CELL_NUMBER

INPUT_KEYBOARD = 0
INPUT_AI = 1

KEYBOARD_DIRECTION = [0, 1, 2, 3, 4] #NOTHING UP DOWN LEFT RIGHT

DIRECTIONS_CLOCKWISE = [0, 1, 2, 3] #UP RIGHT DOWN LEFT 


ITERATION_MAX_WITHOUT_EATING = 500

video = vidmaker.Video(path="output.mp4", fps=30, resolution=(WIN_WIDTH, WIN_HEIGHT))

class SnakeGame:
    def __init__(self, input_type, snake_view=False):
        pygame.init()
        self.snake_view = snake_view
        self.running = True
        self.window = display.set_mode((WIN_WIDTH, WIN_HEIGHT), FLAGS)
        pygame.display.set_caption("SnakeGame")
        self.window.fill(BG_COLOR)
        pygame.mouse.set_visible(0)
        self.score = 0

        self.input_type = input_type
        self.grid = np.zeros((CELL_NUMBER, CELL_NUMBER)) #1 -> head position on the grid  2->body  3->food
        self.snake = []
        self.snake.insert(0, SNAKE_INIT_HEAD_POS)
        self.last_pos = None #used to spawn a body if eat
        self.last_distance = CELL_NUMBER
        self.reward = 0
        self.iteration = 0

        #placement de la case food de manière aléatoire
        self._spawnFood()
        self._updateGridWithNewSnake()

        if(self.input_type == INPUT_KEYBOARD):
            self.direction = KEYBOARD_DIRECTION[1] #start direction = up
        else:
            self.direction = DIRECTIONS_CLOCKWISE[0] #start direction = up

    def playStep(self, dir):

        if(self.input_type == INPUT_KEYBOARD):
            self._playKeyboard(dir)
        else:
            self._playAI(dir)

        self._computeReward() #getting rewarded by getting closer to the food
        #self.reward = 0
        #check for collision
        self._hasCollided()

        self.iteration += 1
        if(self.iteration > len(self.snake)*ITERATION_MAX_WITHOUT_EATING):
            self.running = False
            self.reward = -10

        if(self.running and not self.snake_view):
            self._drawWindow()
        return self.running, self.score, self.reward


    def _playAI(self, dir): # dir = [GO Straight == True, Rotate Left == True, Rotate Right == True]
        self.temp_snake = self.snake.copy()
        self.temp_snake.pop(-1) #delete last to shift all to the previous element position
        self.last_pos = self.snake[len(self.snake)-1].copy()

        if(dir == [0, 0, 1]): #Rotate Right
            self.direction = (self.direction + 1) % len(DIRECTIONS_CLOCKWISE)
        elif(dir == [0, 1, 0]): #Rotate Left
            self.direction -= 1
            if(int(self.direction) < 0):
                self.direction = DIRECTIONS_CLOCKWISE[len(DIRECTIONS_CLOCKWISE) - 1]
        #straight : no changement of direction

        if(self.direction == DIRECTIONS_CLOCKWISE[0]): # UP
            head = self.snake[0].copy()
            head[1] -= 1
            self.temp_snake.insert(0, head)

        elif(self.direction == DIRECTIONS_CLOCKWISE[1]): # RIGHT
            head = self.snake[0].copy()
            head[0] += 1
            self.temp_snake.insert(0, head)

        elif(self.direction == DIRECTIONS_CLOCKWISE[2]): # DOWN
            head = self.snake[0].copy()
            head[1] += 1
            self.temp_snake.insert(0, head)

        elif(self.direction == DIRECTIONS_CLOCKWISE[3]): # LEFT
            head = self.snake[0].copy()
            head[0] -= 1
            self.temp_snake.insert(0, head)



        self.snake = self.temp_snake.copy()
        self._updateGridWithNewSnake()
            

    def _playKeyboard(self, dir):
        self.temp_snake = self.snake.copy()
        self.temp_snake.pop(-1) #delete last to shift all to the previous element position
        self.last_pos = self.snake[len(self.snake)-1].copy()

        if(dir == KEYBOARD_DIRECTION[1] and self.direction != KEYBOARD_DIRECTION[2]): # UP and not 180
            head = self.snake[0].copy()
            head[1] -= 1
            self.temp_snake.insert(0, head)
            self.direction = KEYBOARD_DIRECTION[1]
        elif(dir == KEYBOARD_DIRECTION[2] and self.direction != KEYBOARD_DIRECTION[1]): # DOWN and not 180
            head = self.snake[0].copy()
            head[1] += 1
            self.temp_snake.insert(0, head)
            self.direction = KEYBOARD_DIRECTION[2]
        elif(dir == KEYBOARD_DIRECTION[3] and self.direction != KEYBOARD_DIRECTION[4]): # LEFT and not 180
            head = self.snake[0].copy()
            head[0] -= 1
            self.temp_snake.insert(0, head)
            self.direction = KEYBOARD_DIRECTION[3]
        elif(dir == KEYBOARD_DIRECTION[4] and self.direction != KEYBOARD_DIRECTION[3]): # RIGHT and not 180
            head = self.snake[0].copy()
            head[0] += 1
            self.temp_snake.insert(0, head)
            self.direction = KEYBOARD_DIRECTION[4]
        else: #straight or 180
            if(self.direction == KEYBOARD_DIRECTION[1]): # UP 
                head = self.snake[0].copy()
                head[1] -= 1
                self.temp_snake.insert(0, head)
            elif(self.direction == KEYBOARD_DIRECTION[2]): # DOWN
                head = self.snake[0].copy()
                head[1] += 1
                self.temp_snake.insert(0, head)
            elif(self.direction == KEYBOARD_DIRECTION[3]): # LEFT
                head = self.snake[0].copy()
                head[0] -= 1
                self.temp_snake.insert(0, head)
            elif(self.direction == KEYBOARD_DIRECTION[4]): # RIGHT
                head = self.snake[0].copy()
                head[0] += 1
                self.temp_snake.insert(0, head)

        self.snake = self.temp_snake.copy()
        self._updateGridWithNewSnake()

    def _updateGridWithNewSnake(self):
        self.grid = np.zeros((CELL_NUMBER, CELL_NUMBER))
        try:
            self.grid[self.snake[0][0], self.snake[0][1]] = HEAD
        except IndexError: #end of game : trying to set head at pos 20
            self.running = False

        for i in range(1, len(self.snake)):
            self.grid[self.snake[i][0], self.snake[i][1]] = BODY

        self.grid[self.food[0], self.food[1]] = FOOD

    def _computeReward(self):
        food = self.food #x,y
        head = self.snake[0] #x,y
        food_head_distance = m.sqrt((food[0]-head[0])*(food[0]-head[0]) + (food[1]-head[1])*(food[1]-head[1]))
        if(self.last_distance > food_head_distance and len(self.snake) < 8):
            self.reward = 3
        if(len(self.snake) >= 10): #reward to stay alive
            self.reward = 2
        else:
            self.reward = 0
        self.last_distance = food_head_distance

    def _spawnFood(self):
        food_gen = False
        while(not food_gen):
            self.food = [randint(0,CELL_NUMBER-1), randint(0,CELL_NUMBER-1)]
            if(self.grid[self.food[0], self.food[1]] == 0):
                food_gen = True
                self.grid[self.food[0],self.food[0]] = FOOD

    def _hasCollided(self):
        #stop the game if the snake fill the entire space
        if(len(self.snake) == SNAKE_MAX_SIZE):
            self.running = False
            self.reward = 2000
        
        #Border collision
        head = self.snake[0]
        if(head[0] < 0 or head[0] > CELL_NUMBER-1): #LEFT OR RIGHT
            self.running = False
            self.reward = -10

        if(head[1] < 0 or head[1] > CELL_NUMBER-1): #UP or DOWN
            self.running = False
            self.reward = -10

        #Body collision
        for i in range(1, len(self.snake)):
            if(self.snake[0] == self.snake[i]):
                self.running = False
                self.reward = -10
            
        #Food collision
        if(self.snake[0] == self.food):
            self._eatFood()
            self.reward = 10

    
    def _eatFood(self):
        self.score += 1
        self.snake.append(self.last_pos)
        self._spawnFood()
        self._updateGridWithNewSnake() #reupdate grid
        self.iteration = 0
    
    def _drawWindow(self):
        #clear screen
        self.window.fill(BG_COLOR)
        #draw Snake
        for x in range(self.grid.shape[0]):
            for y in range(self.grid.shape[1]):
                rect = pygame.Rect(x*CELL_SIZE, y*CELL_SIZE, CELL_SIZE, CELL_SIZE)
                if(self.grid[x,y] == HEAD):
                    pygame.draw.rect(self.window, HEAD_COLOR, rect)
                if(self.grid[x,y] == BODY):
                    pygame.draw.rect(self.window, BODY_COLOR, rect)
                if(self.grid[x,y] == FOOD):
                    pygame.draw.rect(self.window, FOOD_COLOR, rect)

        #draw scoreboard
        pygame.draw.line(self.window, TEXT_COLOR, (self.grid.shape[0]*CELL_SIZE,0), (self.grid.shape[0]*CELL_SIZE,WIN_HEIGHT))
        text = f'Score : {self.score}'
        font = pygame.font.SysFont(None, 48)
        img = font.render(text, True, TEXT_COLOR)
        text_width, text_height = font.size(text)
        self.window.blit(img, (WIN_SCOREBOARD_OFFSET + text_width/2, WIN_HEIGHT/2 - text_height/2))
        
        video.update(pygame.surfarray.pixels3d(self.window).swapaxes(0, 1), inverted=True)  # THIS LINE
        pygame.display.flip() #change display buffer to the updated one

    def drawWindowSnakeView(self, grid):
        #clear screen
        self.window.fill(BG_COLOR)
        #draw Snake
        for x in range(grid.shape[0]):
            for y in range(grid.shape[1]):
                rect = pygame.Rect(x*CELL_SIZE, y*CELL_SIZE, CELL_SIZE, CELL_SIZE)
                if(grid[x,y] == HEAD):
                    pygame.draw.rect(self.window, HEAD_COLOR, rect)
                if(grid[x,y] == BODY):
                    pygame.draw.rect(self.window, BODY_COLOR, rect)
                if(grid[x,y] == FOOD):
                    pygame.draw.rect(self.window, FOOD_COLOR, rect)

        #draw scoreboard
        pygame.draw.line(self.window, TEXT_COLOR, (grid.shape[0]*CELL_SIZE,0), (grid.shape[0]*CELL_SIZE,WIN_HEIGHT))
        text = f'Score : {self.score}'
        font = pygame.font.SysFont(None, 48)
        img = font.render(text, True, TEXT_COLOR)
        text_width, text_height = font.size(text)
        self.window.blit(img, (WIN_SCOREBOARD_OFFSET + text_width/2, WIN_HEIGHT/2 - text_height/2))
        pygame.display.flip() #change display buffer to the updated one

    def exportvid(self):
        video.export(verbose=True)