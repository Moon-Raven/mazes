import numpy as np
import matplotlib.pyplot as plt
import random

ROWS = 30
COLS = 50
EXITS = 3

SPACE = 0
WALL = 1

SEEN = 1
UNSEEN = 0

WHITE = 255

def print_info(rows, cols):
    print("Generating maze with {0} rows and {1} columns...".format(rows, cols))

# Returns neighbour indexing tuple from given indexing tuple and desired side
def get_neighbour_ind(index, side):
    row = index[0]
    col = index[1]

    if side == 'L':
        return (row, col-1)
    elif side == 'R':
        return (row, col+1)
    elif side == 'T':
        return (row-1, col)
    elif side == 'B':
        return (row+1, col)
    else:
        print("ERROR: Side {0} is invalid! Exiting...".format(side))

def get_valid_sides(current_index, rows, cols):
    valid_sides = []
    row = current_index[0]
    col = current_index[1]

    if col > 0:
        valid_sides.append('L')

    if col < cols-1:
        valid_sides.append('R')
    
    if row > 0:
        valid_sides.append('T')

    if row < rows-1:
        valid_sides.append('B')

    return valid_sides

# Check if given index has unseen sides
def has_unseen_sides(maze, is_seen, index):
    unseen_sides = get_unseen_sides(maze, is_seen, index)

    if len(unseen_sides) > 0:
        return True
    else:
        return False

def get_unseen_sides(maze, is_seen, current_index):
    max_rows, max_cols = is_seen.shape
    valid_sides = get_valid_sides(current_index, max_rows, max_cols)

    #print(is_seen)
    #print(current_index)
    #print("Valid sides are {0}".format(valid_sides))

    unseen_sides = []
    for side in valid_sides:
        neighbour_ind = get_neighbour_ind(current_index, side)
        #print("{0} of {1} is {2}".format(side, current_index, neighbour_ind))
        if is_seen[neighbour_ind] == UNSEEN:
            unseen_sides.append(side)

    #print("Unseen sides are {0}".format(unseen_sides))
    return unseen_sides

# Returns a random side that has not been seen for the current index cell in maze
def get_random_unseen_side(maze, is_seen, current_index):
    unseen_sides = get_unseen_sides(maze, is_seen, current_index)
    if len(unseen_sides) > 0:
        return random.choice(unseen_sides)
    else:
        return None

def get_opposite_side(side):
    if side == 'L':
        return 'R'
    elif side == 'R':
        return 'L'
    elif side == 'T':        
        return 'B'
    elif side == 'B':
        return 'T'
    else:
        print("Ivalid side {0}, exiting...".format(side))

def remove_wall(maze, current_index, side):
    maze[current_index][side] = SPACE
    neighbour_ind = get_neighbour_ind(current_index, side)
    opposite_side = get_opposite_side(side)
    maze[neighbour_ind][opposite_side] = SPACE

def is_edge(index, max_rows, max_cols):
    row = index[0]
    col = index[1]

    if row == 0 or row == max_rows-1 or col == 0 or col == max_cols-1:
        return True
    else:
        return False

# Generates a random maze with given parameters
def generate_maze(rows, cols, exits):
    print_info(rows, cols)

    maze = np.ones((rows, cols), dtype = [('L', 'i1'), ('R', 'i1'), ('T', 'i1'), ('B', 'i1')])
    is_seen = np.zeros((rows, cols), dtype = np.uint8)

    # Generate a random starting point and mark it as visited
    #start_index = (np.random.randint(rows), np.random.randint(cols))
    start_index = (round(rows/2), round(cols/2))

    trace = [start_index]
    is_seen[start_index] = SEEN
    current_index = start_index

    while True:
        # Select a new random unseen neighbour index
        side = get_random_unseen_side(maze, is_seen, current_index)
        if(side == None):
            break

        # Tear down the wall towards new index
        remove_wall(maze, current_index, side)

        # Move to new index
        current_index = get_neighbour_ind(current_index, side)

        # Mark new index as seen
        is_seen[current_index] = SEEN

        # If all cells are seen, the algorithm is complete
        if not UNSEEN in is_seen:
            break

        # Trace the last seen index with at least one unseen neighbour
        while has_unseen_sides(maze, is_seen, current_index) == False:
            current_index = trace.pop()

        # Add found index to trace
        trace.append(current_index)

        # If reached edge of maze, exit
        #if(is_edge(current_index, rows, cols) == True):
        #    break
    return maze

# Creates an imshow-ready image of the maze
def maze2image(maze):
    rows, cols = maze.shape
    rowsIm, colsIm = rows*2+1, cols*2+1

    # Black image
    image = np.zeros((rowsIm, colsIm, 3), dtype='uint8')

    # Add spaces and remove walls
    for i in range(rows):
        for j in range(cols):

            # Add space
            image[i*2+1, j*2+1] = WHITE

            # Adjust right and bottom walls
            if(maze[i, j]['R'] != WALL):
                image[i*2+1, j*2+1+1] = WHITE

            if(maze[i,j]['B'] != WALL):
                image[i*2+1+1, j*2+1] = WHITE

    return image

# Plots the given maze on screen
def plot_maze(maze):
    image = maze2image(maze)
    plt.imshow(image, interpolation = "None")
    plt.show()

def main():
    maze = generate_maze(ROWS, COLS, EXITS)
    plot_maze(maze)

if __name__ == "__main__":
    main()