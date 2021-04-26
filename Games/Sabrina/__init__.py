#!/usr/bin/env python
# coding: utf-8

# ## PMCS UCT Agent (with 4 ply moves determined)

# ### Game Environment

# In[124]:


import numpy as np
import math
import pandas as pd

# 6 rows by 7 columns => default board size
def get_empty_board(shape=(6, 7)):
    return np.full(shape=shape, fill_value=' ')

# print(get_empty_board())


# In[59]:


# test_board3 = np.array(
#                [[' ', ' ', ' ', ' ', ' ', 'o', 'x'],
#                 [' ', ' ', ' ', ' ', ' ', 'o', 'o'],
#                 [' ', ' ', ' ', 'o', 'x', 'x', 'x'],
#                 ['x', 'x', 'o', 'o', 'x', 'o', 'o'],
#                 ['o', 'x', 'x', 'x', 'o', 'o', 'x'],
#                 ['x', 'x', 'o', 'x', 'x', 'o', 'o']])


# In[73]:


# helper functions to show a simple or color version of the board

import matplotlib.pyplot as plt
from matplotlib import colors

cmap = colors.ListedColormap(["white", "red", "yellow"])
bounds = [0, 1, 2, 3]
norm = colors.BoundaryNorm(bounds, cmap.N)
color_replace = { ' ': 0, 'x': 1, 'o': 2 }

def show_simple_board(board):
    """Prints out the Connect 4 board."""
    print(board)
    
# code adapted from:
# https://stackoverflow.com/questions/43971138/python-plotting-colored-grid-based-on-values
# https://stackoverflow.com/questions/3403973/fast-replacement-of-values-in-a-numpy-array
def show_colored_board(board):
    """Displays the Connect 4 board in a colored grid."""
    colored_board = np.copy(board)
    for k, v in color_replace.items():
        colored_board[board == k] = v
        
    colored_board = colored_board.astype(int)
    
    fig, ax = plt.subplots()
    ax.imshow(colored_board, cmap=cmap, norm=norm)
    
    ax.grid(which='major', axis='both', linestyle='-', color='k', linewidth=2)
    ax.set_xticks(np.arange(-.5, colored_board.shape[1], 1));
    ax.set_yticks(np.arange(-.5, colored_board.shape[0], 1));
        
    ax.tick_params(axis='x', which='both', labelbottom=False)
    ax.tick_params(axis='y', which='both', labelleft=False)
    
    plt.show()
    
# show_colored_board(test_board3)


# In[76]:


# helper function to determine whether a state is a terminal node
# code adapted from https://stackoverflow.com/questions/6313308/get-all-the-diagonals-in-a-matrix-list-of-lists-in-python

import re

def check_win(board):
    """Checks the Connect 4 board for winning states and return one of x, o, d (draw), or n (for next move)."""

    board = np.array(board)
    regex = r'(\S)\1{3}'
        
    rows = [row.astype('|S1').tobytes().decode('utf-8') for row in board]
    cols = [col.astype('|S1').tobytes().decode('utf-8') for col in board.transpose()]
    diags1 = [board[::-1,:].diagonal(i).astype('|S1').tobytes().decode('utf-8') 
              for i in range(-board.shape[0]+1, board.shape[1])]
    diags2 = [board.diagonal(i).astype('|S1').tobytes().decode('utf-8') 
              for i in range(board.shape[1]-1, -board.shape[0], -1)]
    
    for group in [rows, cols, diags1, diags2]:
        for val in group:
            if len(val) < 4:
                continue
                
            matches = re.findall(regex, val)
            if len(matches) > 0:
                return matches[0][0]

    # check for draw
    if(np.sum(board == ' ') == 0):
        return 'd'
    
    return 'n'

# %time print('Win? ' + check_win(test_board3))


# In[79]:


def get_actions(board):
    """Returns possible actions in the form of column numbers that have at least one non-empty slot
    as a vector of indices. Column numbers are zero-indexed."""
    return np.where((np.array(board) == ' ').any(axis=0))[0].tolist()

# %time get_actions(get_empty_board())


# In[81]:


def result(state, player, action):
    """Adds a move to the Connect 4 board for a given player and action."""
    
    col = state[:, action]
    row_idx = np.where(col == ' ')[0][-1]
    col_idx = action
    
    state = state.copy()
    if(state[row_idx][col_idx] != ' '):
        print("Error: Illegal move!")
        
    state[row_idx][col_idx] = player
    
    return state

# show_simple_board(result(get_empty_board(), 'x', 3))


# In[123]:


def other(player):
    if player == 'x': 
        other_player = 'o'
    else: 
        other_player = 'x'
        
    return other_player

def utility(state, player='x'):
    """Checks if a state is a terminal node. Returns the utility if terminal or None if not terminal."""
    
    goal = check_win(state)
    
    if goal == player: return +1 
    if goal == 'd': return 0
    if goal == other(player): return -1
    
    return None # state is not terminal

# print(utility(get_empty_board()))


# In[41]:


# function implementing the environment that calls the agents

def switch_player(player, x, o):
    """Switch player symbol and agent function between turns.
    player is a player symbol and x and o are the players' agent functions."""
    if player == 'x':
        return 'o', o
    else:
        return 'x', x

def play(x, o, N = 100, board_shape=(6, 7)):
    """Play N games. x and o are the players' agent functions."""
    
    results = {'x': 0, 'o': 0, 'd': 0}
    for i in range(N):
        board = get_empty_board(shape=board_shape)
        player, fun = 'x', x
        
        while True:
            a = fun(board, player)
            board = result(board, player, a)
                        
            win = check_win(board)
            if win != 'n':
                results[win] += 1
                if DEBUG >= 1: show_colored_board(board)
                break
            
            player, fun = switch_player(player, x, o)
    return results


# ### Hard-coded four ply moves

# In[83]:


red_move1 = 3
red_moves1 = [3, 1, 5, 3, 1, 1, 3]

yellow_moves1 = [3, 2, 3, 3, 3, 4, 3]
yellow_moves2 = [
    [3,3,3,3,3,3,2],
    [2,1,2,3,2,2,2],
    [3,3,2,3,3,3,3],
    [3,2,4,3,2,4,3],
    [3,3,3,3,4,3,3],
    [4,4,4,3,4,5,4],
    [2,3,3,3,3,3,3]
]

def get_red_second_move_dict():
    move_dict = {}
    for i in range(0, 7):
        board = get_empty_board()
        board = result(board, player='x', action=3)
        board = result(board, player='o', action=i)
        move_dict[hash(board.tobytes())] = red_moves1[i]
        
    return move_dict

def get_yellow_first_move_dict():
    move_dict = {}
    for i in range(0, 7):
        board = get_empty_board()
        board = result(board, player='x', action=3)
        move_dict[hash(board.tobytes())] = yellow_moves1[i]
        
    return move_dict

pre_moves = [(0,3), (1,2), (2,3), (3,3), (4,3), (5,4), (6,3)]

def get_yellow_second_move_dict():
    move_dict = {}
    for i, moves in enumerate(pre_moves):
        board = get_empty_board()
        board = result(board, player='x', action=moves[0])
        board = result(board, player='o', action=moves[1])
        
        for j in range(0, 7):
            current_board = board.copy()
            current_board = result(current_board, player='x', action=j)
            # print(moves, j, yellow_moves2[i][j])
            move_dict[hash(current_board.tobytes())] = yellow_moves2[i][j]
        
    return move_dict


# In[84]:


red_second_move_dict = get_red_second_move_dict()
yellow_first_move_dict = get_yellow_first_move_dict()
yellow_second_move_dict = get_yellow_second_move_dict()


# ### PMCS w/ UCT

# In[87]:


def playout(state, action, player = 'x'):
    """Perform a random playout starting with the given action on the fiven board 
    and return the utility of the finished game."""
    state = result(state, player, action)
    current_player = other(player)
    
    while(True):
        u = utility(state, player)
        if u is not None: 
            return(u)
        
        # we use a random playout policy
        a = np.random.choice(get_actions(state))
        state = result(state, current_player, a)       
        
        current_player = other(current_player)

# %timeit -r1 -n1 print(playout(get_empty_board(), 0))


# In[112]:


def convert_board(board):
    converted_board = board.astype(str)
    converted_board[converted_board == '0'] = ' '
    converted_board[converted_board == '1'] = 'x'
    converted_board[converted_board == '-1'] = 'o'
    return converted_board


# In[118]:


def UCT_depth1_with_first_moves(board, N=100, player=1):
    """Upper Confidence bound applied to Trees for limited tree depth of 1. 
    Simulation budget is N playouts."""
    global DEBUG
    
    board = convert_board(board)
    if player == 1: player = 'x'
    else: player = 'o'
    
    hashed_board = hash(board.tobytes())
    if (player == 'x' and hashed_board == hash(get_empty_board().tobytes())):
        return 3
    if (player == 'x' and hashed_board in red_second_move_dict):
        return red_second_move_dict[hashed_board]
    if (player == 'o' and hashed_board in yellow_first_move_dict):
        return yellow_first_move_dict[hashed_board]
    if (player == 'o' and hashed_board in yellow_second_move_dict):
        return yellow_second_move_dict[hashed_board]
    
    C = math.sqrt(2)
    
    # the tree is 1 action deep
    actions = get_actions(board)
    
    u = [0] * len(actions) # total utility through actions
    n = [0] * len(actions) # number of playouts through actions
    n_parent = 0 # total playouts so far (i.e., number of playouts through parent)
    
    UCB1 = [+math.inf] * len(actions) 
    
    for i in range(N):

        # Select
        action_id = UCB1.index(max(UCB1))
    
        # Expand
        # UTC would expand the tree. We keep the tree at depth 1, essentially performing
        # Pure Monte Carlo search with an added UCB1 selection policy. 
        
        # Simulate
        p = playout(board, actions[action_id], player = player)
    
        # Back-Propagate (i.e., update counts and UCB1)
        u[action_id] += p
        n[action_id] += 1
        n_parent += 1
        
        for action_id in range(len(actions)):
            if n[action_id] > 0:
                UCB1[action_id] = u[action_id]/n[action_id] + C * math.sqrt(math.log(n_parent)/n[action_id])
    
    # return action with largest number of playouts 
    action = actions[n.index(max(n))]
    
    if DEBUG >= 1: 
        print(pd.DataFrame({'action': actions, 
                            'total utility': u, 
                            '# of playouts': n, 
                            'UCB1': UCB1}))
        print()
        print(f"Best action: {action}")
    
    
    return action

# board = get_empty_board() 
# %timeit -n1 -r1 print(UCT_depth1_with_first_move(board))


# In[119]:


def ucb1_10_player_with_first_moves(board, player='x'):
    action = UCT_depth1_with_first_moves(board, N=10, player=player)
    return action

def ucb1_100_player_with_first_moves(board, player='x'):
    action = UCT_depth1_with_first_moves(board, N=100, player=player)
    return action

def ucb1_1000_player_with_first_moves(board, player='x'):
    action = UCT_depth1_with_first_moves(board, N=1000, player=player)
    return action


# ### Tests

# In[122]:


# empty_board = np.full(shape=(6,7), fill_value=0)
# print(empty_board)
# ucb1_100_player_with_first_moves(empty_board)


# In[47]:


# random player

def random_player(board, player = None):
    """Agent that plays Connect 4 by placing its disc in a random column that has at least one non-empty slot."""
    return np.random.choice(get_actions(board))


# In[121]:


# DEBUG = 0
# %timeit -n 1 -r 1 display(play(ucb1_100_player_with_first_moves, random_player, N=10))


# In[49]:


# %timeit -n 1 -r 1 display(play(random_player, ucb1_100_player_with_first_moves, N=10))


# In[50]:


# %timeit -n 1 -r 1 display(play(ucb1_100_player_with_first_moves, ucb1_100_player_with_first_moves, N=10))


# In[ ]:




