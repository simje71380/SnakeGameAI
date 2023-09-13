import pygame
import SnakeGame as SG
import agent
import time

if __name__ == "__main__":
    running = True
    not_break = True
    sg = SG.SnakeGame(SG.INPUT_AI) # use SG.INPUT_AI for AI to play | use SG.INPUT_KEYBOARD if you want to try the game yourself (using arrow keys)
    clock = pygame.time.Clock()

    if(sg.input_type == SG.INPUT_AI):
        agent = agent.Agent(agent.GET_STATE_DANGER_SEE_1)
        agent.load_model()


    while running and not_break:
        if(sg.input_type == SG.INPUT_KEYBOARD):
            direction = SG.KEYBOARD_DIRECTION[0]

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                not_break = False
            if event.type == pygame.KEYDOWN and sg.input_type == SG.INPUT_KEYBOARD:
                if event.key == pygame.K_q:
                    print("exit : Q pressed")
                    not_break = False
                if event.key == pygame.K_UP:
                    direction = SG.KEYBOARD_DIRECTION[1]
                if event.key == pygame.K_DOWN:
                    direction = SG.KEYBOARD_DIRECTION[2]
                if event.key == pygame.K_LEFT:
                    direction = SG.KEYBOARD_DIRECTION[3]
                if event.key == pygame.K_RIGHT:
                    direction = SG.KEYBOARD_DIRECTION[4]
            if event.type == pygame.KEYDOWN and sg.input_type == SG.INPUT_AI:
                if event.key == pygame.K_q:
                    print("exit : Q pressed")
                    not_break = False

        if(sg.input_type == SG.INPUT_AI):
            state = agent.get_state(sg)
            direction = agent.get_action_NN(state)


        #print(f"state : {agent.get_state(sg)}")
        running, score, reward = sg.playStep(direction)
        #print(f"play stil alive : {running}")
        if(sg.input_type == SG.INPUT_AI):
            clock.tick(SG.REFRESH_RATE_AI)
        else:
            clock.tick(SG.REFRESH_RATE)


    sg.exportvid()
    print(f"Game ended | score : {score}")
    time.sleep(1)