import numpy as np
import matplotlib.pyplot as plt

SPACE = 0
WALL = 255

SEEN = 1
UNSEEN = 0

def print_info(rows, cols):
    print("Generating maze with {0} rows and {1} columns...".format(rows, cols))

def generate_maze(rows, cols, exits):
    print_info(rows, cols)
    maze = np.zeros((rows, cols), dtype = np.uint8)
    visited = np.zeros((rows, cols), dtype = np.uint8)

    start_cell = [np.random.randint(rows), np.random.randint(cols)]
    print(start_cell)
    current_cell = start_cell

    maze[start_cell] = WALL

    i = 0
    while i < 20:
        maze [current_cell] = WALL

        direction = np.random.randint(4)
        if direction == 1:
            current_cell = ('''
    return maze

def plot_maze(maze):
    plt.imshow(maze, cmap = "Greys", interpolation = "None")
    plt.show()

def main():
    maze = generate_maze(50, 50, 3)
    plot_maze(maze)

if __name__ == "__main__":
    main()