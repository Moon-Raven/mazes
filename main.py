import numpy as np
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import random
from collections import deque

ROWS = 30
COLS = 50
EXITS = 1

SPACE = 0
WALL = 1

SEEN = 1
UNSEEN = 0

WHITE = 255
DARK_GREEN = (0, 150, 0)
BLUE = (50, 120, 255)
RED = (255, 0, 0)

RED_TUPLE = (1,0,0)
DARK_GREEN_TUPLE = (0, 150/255, 0)
BLUE_TUPLE = (50/255, 120/255, 1)

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

# Get all valid sides of given index (edge cells have less valid sides)
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

# Returns all sides of given index that have not been seen yet
def get_unseen_sides(maze, is_seen, current_index):
    max_rows, max_cols = is_seen.shape
    valid_sides = get_valid_sides(current_index, max_rows, max_cols)

    unseen_sides = []
    for side in valid_sides:
        neighbour_ind = get_neighbour_ind(current_index, side)
        if is_seen[neighbour_ind] == UNSEEN:
            unseen_sides.append(side)

    return unseen_sides

# Returns a random side that has not been seen for the current index cell in maze
def get_random_unseen_side(maze, is_seen, current_index):
    unseen_sides = get_unseen_sides(maze, is_seen, current_index)
    if len(unseen_sides) > 0:
        return random.choice(unseen_sides)
    else:
        return None

def get_connected_unseen_neighbours(maze, is_seen, index):
    unseen_sides = get_unseen_sides(maze, is_seen, index)
    #print("Maze: " + str(maze[index]))
    #print("Unseen sides: {0}".format(unseen_sides))
    connected_unseen_sides = [side for side in unseen_sides if maze[index][side] == SPACE]
    #print("Connected unseen sides: {0}".format(connected_unseen_sides))
    connected_unseen_neighbours = [get_neighbour_ind(index, side) for side in connected_unseen_sides]
    #print("Connected unseen neighbours: {0}".format(connected_unseen_neighbours))
    return connected_unseen_neighbours


# Gets a list of unseen neighbour indexes of given index
def get_unseen_neighbours(maze, is_seen, index):
    unseen_sides = get_unseen_sides(maze, is_seen, index)
    unseen_neighbours = [get_neighbour_ind(index, side) for side in unseen_sides]
    return unseen_neighbours

# Returns opposite side ('L' <-> 'R', 'T' <-> 'B')
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

# Removes the desired wall from the given maze
def remove_wall(maze, current_index, side):
    maze[current_index][side] = SPACE
    neighbour_ind = get_neighbour_ind(current_index, side)
    opposite_side = get_opposite_side(side)
    maze[neighbour_ind][opposite_side] = SPACE

# Checks if given cell is edge cell of the maze
def is_edge(index, max_rows, max_cols):
    row = index[0]
    col = index[1]

    if row == 0 or row == max_rows-1 or col == 0 or col == max_cols-1:
        return True
    else:
        return False

# Prepares a maze with all walls occupied
def generate_empty_maze(rows, cols):
    maze = np.ones((rows, cols), dtype = [('L', 'i1'), ('R', 'i1'), ('T', 'i1'), ('B', 'i1'), ('exit', 'b')])
    maze[:]['exit'] = False
    return maze

# Gets all edge cells of a maze
def get_edge_cells(maze):
    rows, cols = maze.shape

    rr = np.arange(rows)
    cc = np.arange(cols)

    edges = set([(a, b) for a in rr for b in [0, cols-1]] + \
                [(a, b) for a in [0, rows-1] for b in cc])

    return edges

# Adds the desired number of exits to the given maze
def add_exits(maze, exit_num):
    edge_cells = get_edge_cells(maze)
    chosen_exits = random.sample(edge_cells, exit_num)

    for exit in chosen_exits:
        maze[exit]['exit'] = True

# Get all walls from the given index
def get_walls_of_index(index, rows, cols):
    sides = get_valid_sides(index, rows, cols)
    walls = [(index, side) for side in sides]
    return walls

# Add the given wall to the list of walls
def add_walls_to_list(index, wall_list, rows, cols):
    walls = get_walls_of_index(index, rows, cols)
    wall_list += walls

# Get opposite cell of the wall
def get_opposite_cell(wall):
    return get_neighbour_ind(wall[0], wall[1])

# Check if both cells touching the wall have been seen
def both_cells_seen(wall, is_seen):
    ind = wall[0]
    ind2 = get_opposite_cell(wall)

    if is_seen[ind] == SEEN and is_seen[ind2] == SEEN:
        return True
    else:
        return False
 
# Removes the given wall from the maze
def remove_wall_prim(maze, wall):
    remove_wall(maze, wall[0], wall[1])

# Generates a random maze with given parameters using randomized Prim algorithm
def generate_maze_prim(rows, cols, exits):
    maze = generate_empty_maze(rows, cols)
    is_seen = np.zeros((rows, cols), dtype = np.uint8)

    # Generate a random starting point and mark it as visited
    start_index = (np.random.randint(rows), np.random.randint(cols))

    is_seen[start_index] = SEEN
    current_index = start_index

    wall_list = []
    add_walls_to_list(current_index, wall_list, rows, cols)

    while len(wall_list) > 0:
        wall = random.choice(wall_list)

        if not both_cells_seen(wall, is_seen):
            remove_wall_prim(maze, wall)
            neighbour_ind = get_opposite_cell(wall)
            is_seen[neighbour_ind] = SEEN
            add_walls_to_list(neighbour_ind, wall_list, rows, cols)

        wall_list.remove(wall)

    add_exits(maze, exits)

    return maze

# Generates a random maze with given parameters using depth-first search
def generate_maze_dfs(rows, cols, exits):
    maze = generate_empty_maze(rows, cols)
    is_seen = np.zeros((rows, cols), dtype = np.uint8)

    # Generate a random starting point and mark it as visited
    start_index = (np.random.randint(rows), np.random.randint(cols))

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

    # Add the desired number of exits
    add_exits(maze, exits)

    return maze

# Adds the exit to the given image
def add_exit_to_image(image, i, j, rows, cols):

    if i == 0:
        image[i*2+1-1, j*2+1] = DARK_GREEN
    elif j == 0:
        image[i*2+1, j*2+1-1] = DARK_GREEN
    elif j == cols-1:
        image[i*2+1, j*2+1+1] = DARK_GREEN
    else:
        image[i*2+1+1, j*2+1] = DARK_GREEN

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

            if maze[i,j]['exit'] == True:
                add_exit_to_image(image, i, j, rows, cols)

    return image

# Returns a list of indexes that are exits of the maze
def get_exits(maze):
    exits = []
    it = np.nditer(maze, flags=['multi_index'])
    while not it.finished:
        if it[0]['exit'] == True:
            exits.append(it.multi_index)
        it.iternext()

    return exits

# Returns a list of nodes which lead from root to end_node
def trace_to_root(graph, end_node):
    path = []

    current = end_node
    while current:
        path.append(current)
        current = graph[current]

    path.reverse()
    return path

#Finds the shortest path from start to end using breadth-first search
def shortest_path(start, end, maze):
    rows, cols = maze.shape
    is_seen = np.zeros((rows, cols), dtype = np.uint8)

    result_graph = {start:None}
    q = deque([start])

    while q:
        # Fetch next node to examine and mark it as seen
        current = q.popleft()
        is_seen[current] = SEEN

        # If it is the end we are searching for, terminate
        if current == end:
            break

        # Find all children and enqueue them
        children = get_connected_unseen_neighbours(maze, is_seen, current)
        for child in children:
            q.append(child)
            result_graph[child] = current

    path = trace_to_root(result_graph, end)
    return path


# Solves the maze using breadth-first search
def solve_maze(maze, start_index):
    exits = get_exits(maze)
    solutions = [shortest_path(start_index, end, maze) for end in exits]
    return solutions

# Returns the pixel coordinates of the space between two cells
def get_pixel_between_indexes(ind1, ind2):
    i1, j1 = ind1
    i2, j2 = ind2
    
    res_i = round((i1+i2)/2*2+1)
    res_j = round((j1+j2)/2*2+1)

    return (res_i, res_j)

# Adds a single solution to the image of the maze
def add_solution_to_image(image, solution):
    start = solution[0]
    image[start[0]*2+1, start[1]*2+1] = RED
    prev = start

    for index in solution[1:]:
        i, j = index
        image[i*2+1, j*2+1] = BLUE

        if prev != None:
            pixel = get_pixel_between_indexes(index, prev)
            image[pixel] = BLUE

        prev = index

# Adds all solutions to the image of the maze
def add_solutions_to_image(image, solutions):
    for solution in solutions:
        add_solution_to_image(image, solution)

# Plots the given maze on screen
def plot_maze(maze, solutions):
    image = maze2image(maze)
    add_solutions_to_image(image, solutions)
    plt.imshow(image, interpolation = "None")
    red_patch = mpatches.Patch(color=RED_TUPLE, label='Start node')
    dark_green_patch = mpatches.Patch(color=DARK_GREEN_TUPLE, label='Exits')
    blue_patch = mpatches.Patch(color=BLUE_TUPLE, label='Solution')
    plt.legend(handles=[red_patch, dark_green_patch, blue_patch])
    plt.show()

# Prints lengths of solutions
def print_solutions(solutions):
    print("Printing solutions info:")
    for solution in solutions:
        start = solution[0]
        end = solution[-1]
        l = len(solution)

        print("\tLength of path from {0} to {1} is {2}".format(start, end, l))

# Prompts user input for maze configuration
def get_user_input():
    try:
        rows = int(input("Enter number of rows: "))
        cols = int(input("Enter numbers of columns: "))
        exits = int(input("Enter numbers of exits: "))
        method = input("Enter desired method(either 'dfs' or 'prim'): ")   

        if method != 'dfs' and method != 'prim':
            print("Invalid selected algorithm! Terminating...")
            exit()

        s = input("Enter start coordinates (eg. 2,5): ")
        s = s.strip()
        s = s.split(",")
        start = (int(s[0].strip()), int(s[1].strip()))

        if start[0] < 0 or start[0] >= rows or start[1] < 0 or start[1] >= cols:
            print("Start coordinates out of range! Terminating...")
            exit()
        
    except:
        print("Error during input!")
        exit()

    print()
    return rows, cols, exits, method, start

def main():
    # Get user input
    rows, cols, exits, method, start = get_user_input()

    # Generate maze using the selected algorithm
    if method == 'dfs':
        maze = generate_maze_dfs(rows, cols, exits)
    elif method == 'prim':
        maze = generate_maze_prim(rows, cols, exits)

    # Solve the maze using bfs
    solutions = solve_maze(maze, start)

    # Print solution info and plot the maze with solutions
    print_solutions(solutions)
    plot_maze(maze, solutions)

if __name__ == "__main__":
    main()