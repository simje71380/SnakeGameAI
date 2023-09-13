from agent import train
import sys
import agent as Ag

def main(argv):
    if(argv[0] == '1_tile'):
        a_mode = Ag.GET_STATE_DANGER_SEE_1
    if(argv[0] == '2_tile'):
        a_mode = Ag.GET_STATE_DANGER_SEE_2
    if(argv[0] == '3_tile'):
        a_mode = Ag.GET_STATE_DANGER_3_AROUND
    if(argv[0] == 'full'):
        a_mode = Ag.GET_STATE_GRID
    else:
        print("\n\nERROR: specify the observation mode : 1_tile | 2_tile | 3_tile | full\n--> run: python train.py 1_tile\nor run: python train.py  full ...")
        return
    
    train(mode=a_mode)

if __name__ == '__main__':
    main(sys.argv[1:])
