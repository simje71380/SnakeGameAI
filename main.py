import pygame
import SnakeGame as SG
import agent as Ag
import time
import sys

def main(argv):
    if(argv[0] == 'human'):
        mode = SG.INPUT_KEYBOARD
    elif(argv[0] == 'AI'):
        mode = SG.INPUT_AI
        if(argv[1] == '1_tile'):
            a_mode = Ag.GET_STATE_DANGER_SEE_1
        if(argv[1] == '2_tile'):
            a_mode = Ag.GET_STATE_DANGER_SEE_2
        if(argv[1] == '3_tile'):
            a_mode = Ag.GET_STATE_DANGER_3_AROUND
        if(argv[1] == 'full'):
            a_mode = Ag.GET_STATE_GRID
        else:
            print("\n\nERROR: specify the observation mode : 1_tile | 2_tile | 3_tile | full\n--> run: python main.py AI 1_tile\nor run: python main.py AI full ...")
    else:
        print("\n\nERROR: specify game mode : human or AI\n--> run: python main.py human\nor run: python main.py AI observation_mode")
        return 1

    running = True
    not_break = True
    sg = SG.SnakeGame(mode)
    clock = pygame.time.Clock()

    if(sg.input_type == SG.INPUT_AI):
        agent = Ag.Agent(mode = a_mode)
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

if __name__ == "__main__":
    main(sys.argv[1:])