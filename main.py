import pygame
import SnakeGame as SG


if __name__ == "__main__":
    running = True
    not_break = True
    sg = SG.SnakeGame(SG.INPUT_KEYBOARD)
    clock = pygame.time.Clock()


    while running and not_break:
        direction = SG.KEYBOARD_DIRECTION[0]
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                not_break = False
            if event.type == pygame.KEYDOWN:
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

        running, score = sg.playStep(direction)
        clock.tick(SG.REFRESH_RATE)
    
    print(f"You lost score : {score}")