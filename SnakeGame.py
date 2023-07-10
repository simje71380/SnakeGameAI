import pygame
from pygame import *
import numpy as np
from random import *


#Window properties
WIN_WIDTH = 960
WIN_HEIGHT = 720
WIN_SCOREBOARD_OFFSET = WIN_WIDTH - (WIN_WIDTH - WIN_HEIGHT)

CELL_NUMBER = 20
CELL_SIZE = int(WIN_HEIGHT/CELL_NUMBER)

FLAGS = DOUBLEBUF


#Game properties
HEAD_COLOR = (0, 128, 0)
BODY_COLOR = (0, 255, 0)
FOOD_COLOR = (255, 0, 0)
BG_COLOR = (0, 0, 0)
TEXT_COLOR = (255, 255, 255)
REFRESH_RATE = 8
SNAKE_INIT_HEAD_POS = [int(CELL_NUMBER/2), int(CELL_NUMBER/2)]

HEAD = 1
BODY = 2
FOOD = 3

SNAKE_MAX_SIZE = CELL_NUMBER * CELL_NUMBER

INPUT_KEYBOARD = 0
INPUT_AI = 1

KEYBOARD_DIRECTION = [0, 1, 2, 3, 4] #NOTHING UP DOWN LEFT RIGHT

class SnakeGame:
    def __init__(self, input_type):
        pygame.init()
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
        self.grid[SNAKE_INIT_HEAD_POS[0], SNAKE_INIT_HEAD_POS[1]] = HEAD
        self.last_pos = None #used to spawn a body if eat

        #placement de la case food de manière aléatoire
        self._spawnFood()

        if(input_type == INPUT_KEYBOARD):
            self.direction = KEYBOARD_DIRECTION[2] #start direction = down

    def playStep(self, dir):
        if(self.input_type == INPUT_KEYBOARD):
            self._playKeyboard(dir)

        #check for collision
        self._hasCollided()

        if(self.running):
            self._drawWindow()
        return self.running, self.score

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
        
        #Border collision
        head = self.snake[0]
        if(head[0] < 0 or head[0] > CELL_NUMBER-1): #LEFT OR RIGHT
            self.running = False

        if(head[1] < 0 or head[1] > CELL_NUMBER-1): #UP or DOWN
            self.running = False

        #Body collision
        for i in range(1, len(self.snake)):
            if(self.snake[0] == self.snake[i]):
                self.running = False
            
        #Food collision
        if(self.snake[0] == self.food):
            self._eatFood()

    
    def _eatFood(self):
        self.score += 1
        self.snake.append(self.last_pos)
        self._spawnFood()
        self._updateGridWithNewSnake() #reupdate grid
    
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
        pygame.display.flip() #change display buffer to the updated one