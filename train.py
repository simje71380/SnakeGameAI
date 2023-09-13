from agent import train

#snake vision modes
GET_STATE_GRID = 0
GET_STATE_DANGER_SEE_1 = 1
GET_STATE_DANGER_SEE_2 = 2
GET_STATE_DANGER_3_AROUND = 3

if __name__ == '__main__':
    train(mode=GET_STATE_GRID)