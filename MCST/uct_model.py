#!/usr/bin/env python
# encoding: utf-8
"""

This file is a part of DDZ AI.

Copyright (C) 2018 JJ zhangqi02@mail.jj.cn

"""


import copy
import math
import sys
import time
import config
from core import rule
from engine import move


# PUCT = 1 / math.sqrt(2.0)
PUCT = 5
DISPLAY = 1
EFFORT_BIAS = 0.8
EFFORT = 1000
FIRST_PLAY = 5
SECOND_PLAY = 5
TERMINAL = 2


class Node:
    def __init__(self):
        self.parent = None
        self.children = []

        self.visit = 0  # N
        self.accu_value = 0.0  # W
        self.avg_value = 0.0  # Q

        self.state = None  # mapping between game state and tree node
        self.is_leaf = True  # New expanded node should be a leaf.

        self.action = ''  # move with rank and suit, e.g., D4
        self.p = 0.0  # prior probabilities from policy network


def tree_policy(node) -> Node:
    while not node.state.game_is_over():
        if node.is_leaf:
            policy_expand(node)
            return node
        else:
            node = ucb_select_child(node)

    return node


def ucb_select_child(node) -> Node:
    best_score = -sys.maxsize
    best_child_node = None
    to_move = node.state.player_list[node.state.about_to_play].role  # landlord or peasant

    for child_node in node.children:
        avg_value = child_node.avg_value
        q = avg_value if to_move == 'Landlord' else -0.5 * avg_value
        u = child_node.p * math.sqrt(node.visit) / (1 + child_node.visit)  # avoid zero denominator

        score = q + PUCT * u  # UCB algorithms
        if score > best_score:
            best_child_node = child_node
            best_score = score

    return best_child_node


def policy_expand(node) -> None:
    node.is_leaf = False

    current_state = node.state
    top3_flag = True if node.parent is None else False
    move_prob_dict, is_legal = move.policy_eval(current_state, top3_flag)

    if not is_legal:
        all_legal_set = all_legal_moves(node)
        for combo in all_legal_set:
            construct_node(node, combo.content, 1.0)
    else:
        for next_move, prob in move_prob_dict.items():
            construct_node(node, next_move, prob)


def default_policy(node) -> int:
    state_copy = copy.deepcopy(node.state)
    play_with_random_choices(state_copy)
    cal_score(state_copy)
    for player in state_copy.player_list:
        if player.role == 'Landlord':
            return player.score


def backup(node, value: int) -> None:
    while node is not None:  # The root node doesn't have parent.
        node.visit += 1
        node.accu_value += value
        node.avg_value = node.accu_value / node.visit

        node = node.parent


def uct_search(node, time_start: float) -> tuple:
    # Run as much as possible under the computation budget.
    combo = node.state.combo
    config.THINKING_TIME = FIRST_PLAY if combo.combo_type == 'First' else SECOND_PLAY
    for i in range(sys.maxsize):
        time_end = time.time()
        if time_end - time_start >= config.THINKING_TIME - config.LAG_BUFFER:
            # logging.info(f'[uct_search] end time >={config.THINKING_TIME - config.LAG_BUFFER},
            # using : {time_end - time_start}, run times：[{i}]')
            break

        # core codes
        leaf_node = tree_policy(node)
        if not node.children:
            return False, ''
        value = default_policy(leaf_node)
        backup(leaf_node, value)

        if i + 1 >= DISPLAY and (i + 1) % DISPLAY == 0 and not config.SILENCE:
            print_result(node)

        # if i + 1 >= EFFORT:
        if early_stop(node):
            return True, 'EARLY_STOP'
    if len(node.children) == 0:
        return False, ''

    return True, 'UCT_END'


def early_stop(node) -> bool:
    if len(node.children) == 1:
        return True

    total_visits = node.visit
    best_child = choose_best(node)
    effort = best_child.visit / node.visit

    if total_visits >= EFFORT:
        if effort >= EFFORT_BIAS:
            return True

    return False


def play_with_random_choices(state_copy) -> None:
    while not state_copy.game_is_over():
        if state_copy.pass_num == 2:
            state_copy.combo = rule.Combo('First', 0, '', 0)

        next_move = move.genmove_random(state_copy)
        if not rule.play_a_move(state_copy, next_move.split(' ')):
            print('Illegal move.')


def cal_score(state) -> None:
    if state.summer() or state.anti_summer():
        state.multi *= 2
        state.score *= 2

    for player in state.player_list:
        if player.role == state.winner:
            if player.role == 'Landlord':
                player.score += 2 * state.score
            else:
                player.score += state.score
        else:
            if player.role == 'Landlord':
                player.score -= 2 * state.score
            else:
                player.score -= state.score


def choose_best(node) -> Node:
    # Choose the most visited node from the root position.
    max_visit = -sys.maxsize
    best_child = None

    for child in node.children:
        if child.visit > max_visit:
            best_child = child
            max_visit = child.visit

    return best_child


def print_result(root) -> None:
    print('After {} simulations...'.format(root.visit))

    to_move = root.state.player_list[root.state.about_to_play].role
    node_list = sorted(root.children, key=lambda x: x.visit, reverse=True)
    for node in node_list:
        if to_move == 'Landlord':
            value = node.avg_value
        else:
            if node.avg_value == 0:
                value = 0.0
            else:
                value = -0.5 * node.avg_value

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
        print(' ' + best_child.action, end=' |')
        current_root_copy = best_child
    print()

    best_child = choose_best(root)
    if to_move == 'Landlord':
        win_rate = best_child.avg_value
    else:
        win_rate = -0.5 * best_child.avg_value
    print('value: {:.1f}'.format(win_rate))

    print()


def construct_node(node, next_move: str, prob: float) -> None:
    sub_node = Node()

    current_state = node.state
    hand = current_state.player_list[current_state.about_to_play].hand
    next_move_with_suit = allocate_suit(next_move, hand)
    sub_node.action = next_move_with_suit
    sub_node.state = get_next_state(node.state, next_move_with_suit)
    sub_node.p = prob
    sub_node.parent = node

    node.children.append(sub_node)


def allocate_suit(rank_str: str, hand: list) -> str:
    hand_str_list = []
    if rank_str == 'Pass':
        return 'Pass'

    for card in sorted(hand, key=lambda c: c.level, reverse=True):
        if card.rank == 'RJ' or card.rank == 'BJ':
            hand_str_list.append(card.rank)
        else:
            card_str = card.suit[0] + card.rank
            hand_str_list.append(card_str)

    card_str_list = []
    if rank_str != 'Pass':
        for rank in rank_str.split(' '):
            if rank == 'RJ' or rank == 'BJ':
                card_str_list.append(rank)
                hand_str_list.remove(rank)
                continue
            hand_str_list_copy = hand_str_list.copy()
            for card_str in hand_str_list_copy:
                if card_str[-1] == rank and card_str != 'RJ' and card_str != 'BJ':
                    card_str_list.append(card_str)
                    hand_str_list.remove(card_str)
                    break

    return ' '.join(card_str_list)


def get_next_state(current_state, next_move_with_suit: str) -> Node:
    current_state_copy = copy.deepcopy(current_state)
    if not rule.play_a_move(current_state_copy, next_move_with_suit.split(' ')):
        print('Illegal move.')

    if current_state_copy.pass_num == 2:
        current_state_copy.combo = rule.Combo('First', 0, '', 0)

    return current_state_copy


def all_legal_moves(node) -> set:
    current_state = node.state
    who_to_play = current_state.player_list[current_state.about_to_play]
    hand_copy = who_to_play.hand.copy()
    legal_move_set = set()
    legal_moves = rule.get_legal_moves(current_state.combo, hand_copy)
    for combo_type in legal_moves:
        legal_move_set.update(legal_moves[combo_type])

    return legal_move_set


def genmove_uct(game) -> tuple:
    time_start = time.time()
    config.PURE_NN = False  # correlated with display

    root = Node()
    root.state = game
    success, note = uct_search(root, time_start)
    if not success:
        return '', 'ILLEGAL'

    best_node = choose_best(root)

    if not config.SILENCE:
        print_result(root)

    return best_node.action, note
