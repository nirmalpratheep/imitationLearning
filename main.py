import sys
from game.curriculum_game import run

if __name__ == "__main__":
    level = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    run(start_track=level)
