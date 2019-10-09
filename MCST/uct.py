# -*- coding: utf-8 -*-
# @Time    : 2019/8/5 13:55
# @Author  : hanyu
# @mail    ：hanyu01@mail.jj.cn

import copy
import logging
import math
import sys
import time
import random
# PUCT = 1 / math.sqrt(2.0)
AVAILABLE_CHOICES = [0, 1, 2, 3]

PUCT = 5
DISPLAY = 1
EFFORT_BIAS = 0.8
EFFORT = 800
FIRST_PLAY = 5
SECOND_PLAY = 5
TERMINAL = 2
SILENCE = False

class gameOb():
    def __init__(self, coordinates, size, intensity, channel, reward, name):
        self.x = coordinates[0]
        self.y = coordinates[1]
        self.size = size
        self.intensity = intensity
        self.channel = channel
        self.reward = reward
        self.name = name


class Node:
    def __init__(self):
        self.parent = None
        self.children = []

        self.visit = 0 # N
        self.acc_value = 0.0 # W
        self.avg_value = 0.0 # Q

        self.state = None # mapping between game state nd tree node
        self.is_leaf = True # new expanded node should be a leaf node

        self.action = '' # move with rank and suit
        self.p = 0.0 # prior probabilities from policy network

def uct_search(node, time_start: float) -> tuple:
    # Run as much as possible under the computation budget

    for i in range(sys.maxsize):
        time_end = time.time()
        if time_end - time_start >= 1000:
            logging.info(f'[uct_search] end time >={10}, '
                         f'using: {time_end - time_start}, run steps: [{i}]')
            break

        # UCT core code
        leaf_node = tree_policy(node)
        if not node.children:
            return False, ''
        value = default_policy(leaf_node)
        backup(leaf_node, value)

        if i + 1 >= DISPLAY and (i + 1) % DISPLAY == 0 and not SILENCE:
            print_result(node)
        # if i + 1 >= DISPLAY and (i + 1) % DISPLAY == 0:
        #     print_result(node)

        if i + 1 >= EFFORT:
            if early_stop(node):
                return True, 'EARLY_STOP'

    if len(node.children) == 0:
        return False, ''
    return True, 'UCT_END'

def early_stop(node):
    if len(node.children) == 1:
        return True

    # total_visits = node.visit
    best_child = choose_best(node)
    effort = best_child.visit / node.visit

    if effort >= EFFORT_BIAS:
        return True
    return False

def choose_best(node):
    # choose the most visited node from the root position
    max_visit = -sys.maxsize
    best_child = None

    for child in node.children:
        if child.visit > max_visit:
            best_child = child
            max_visit = child.visit
    return best_child

def backup(node, value):
    while node is not None: # root node does not have parent node
        node.visit += 1
        node.acc_value += value
        node.avg_value = node.acc_value / node.visit
        node = node.parent


def default_policy(node):
    state_copy = copy.deepcopy(node.state)
    play_with_random_choices(state_copy)
    state_copy.cal_score()
    return state_copy.score



def play_with_random_choices(state):
    while not state.game_is_over():
        # todo using rollout to get next_move
        next_move = random.choice([choice for choice in AVAILABLE_CHOICES])
        hero_pos = state.moveChar(next_move)


def tree_policy(node: Node) -> Node:
    while not node.state.game_is_over():
        if node.is_leaf:
            policy_expand(node)
            return node
        else:
            node = ucb_select_child(node)
    return node

def ucb_select_child(node):
    best_score = -sys.maxsize
    best_child_node = None

    for child_node in node.children:
        avg_value = child_node.avg_value
        q = avg_value
        u = child_node.p * math.sqrt(node.visit) / (1 + child_node.visit)

        score = q + PUCT * u # ucb algorithm
        if score > best_score:
            best_child_node = child_node
            best_score = score
    return best_child_node


def policy_expand(node: Node) -> None:
    node.is_leaf = False

    current_state = node.state
    move_list = [0, 1, 2, 3]
    for move in move_list:
        construct_node(node, move, 0.25)

def construct_node(node, next_move, prob: float) -> None:
    sub_node = Node()
    sub_node.action = next_move
    sub_node.state = get_next_state(node.state, next_move)
    sub_node.p = prob
    sub_node.parent = node
    node.children.append(sub_node)

def get_next_state(current_state, next_move: int):
    current_state_copy = copy.deepcopy(current_state)
    current_state_copy.moveChar(next_move)
    return current_state_copy


def print_result(root) -> None:
    print('After {} simulations...'.format(root.visit))

    node_list = sorted(root.children, key=lambda x: x.visit, reverse=True)
    for node in node_list:
        value = node.avg_value

        prob = 100.0 * node.p
        visit = node.visit
        if root.visit > 1:
            effort = 100.0 * node.visit / (root.visit - 1)  # The root has been expanded once.
        else:
            effort = 0.0

        print('next move: {}  |  prob: {:.1f}%  |  visit: {}  |  value: {:.1f}  |  effort: {:.1f}%'.format(
            node.action, prob, visit, value, effort))

        if node_list.index(node) >= 10:  # Display top-10 at most
            break

    print('pv: |', end='')  # principle variant
    current_root_copy = copy.deepcopy(root)
    while not current_root_copy.is_leaf:
        best_child = choose_best(current_root_copy)
        print(' ' + str(best_child.action), end=' |')
        current_root_copy = best_child
    print()

    best_child = choose_best(root)
    win_rate = best_child.avg_value
    print('value: {:.1f}'.format(win_rate))

def genmove_uct(game) -> tuple:
    time_start = time.time()

    root = Node()
    root.state = game
    success, note = uct_search(root, time_start)
    if not success:
        return '', 'FAIL'
    best_node = choose_best(root)

    if not SILENCE:
        print_result(root)

    return best_node.action, note

from recommend_system_practise.MCST.gridworld import gameEnv
if __name__ == '__main__':
    game = gameEnv()
    game.reset()
    move = genmove_uct(game)