# myTeam.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to
# http://inst.eecs.berkeley.edu/~cs188/pacman/pacman.html
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).
from functools import wraps

from captureAgents import CaptureAgent
from _player_state import PlayerState
from game import AgentState, Directions

# extension
PlayerState.extend_agent_state(AgentState)


#################
# Team creation #
#################

def createTeam(firstIndex, secondIndex, isRed,
               first='MainAgent', second='MainAgent'):
    return [eval(first)(firstIndex), eval(second)(secondIndex)]


################
# Util Classes #
################


class FeatureSet:

    def __init__(self, groups=None):
        self.fns = dict()
        self.ws = dict()
        self.vs = dict()
        self.names = list()
        if groups is not None:
            self.group_append(groups)

    def append(self, w, fn):
        n = fn.__name__
        self.names.append(n)
        self.fns[n] = fn
        self.ws[n] = w

    def group_append(self, groups):
        for (w, fn) in groups:
            self.append(w, fn)

    def calc(self, player):
        for n in self.names:
            self.vs[n] = self.fns[n](player)

    def eval(self, player):
        self.calc(player)
        values = [self.vs[n] * self.ws[n] if self.vs[n] is not None else 0 for n in self.names]
        # print self.vs
        # print self.ws
        return sum([0] + values)


class StateMachine:

    def __init__(self):
        self.state_dict = dict()
        self.state = None
        self.state_name = None

    @property
    def state_names(self):
        reserved_names = ['state_names', 'state_dict', 'act', 'state', 'state_name', 'default', 'reset']
        return [name for name in dir(self) if not name.startswith('_') and name not in reserved_names]

    def act(self, *args):
        results = [(getattr(self, name)(*args), name) for name in self.state_names]
        for (fn, need_transfer, priority), name in sorted(results, key=lambda x: x[0][2]):
            if need_transfer:
                # print "-> " + name
                self._transfer(fn, name)

    def _transfer(self, fn, name):
        # print "-> " + name
        self.state_name = name
        if name not in self.state_dict:
            self.state_dict[name] = fn()
        self.state = self.state_dict[name]

    def reset(self):
        self.state = None
        self.state_name = None
        if 'default' in dir(self):
            a_class, name = getattr(self, 'default')()
            self._transfer(a_class, name)


def state(fn, priority=0):
    def transfer_decorator(transfer_fn):
        @wraps(transfer_fn)
        def wrapper(*args, **kwargs):
            need_transfer = transfer_fn(*args, **kwargs)
            return fn, need_transfer, priority

        return wrapper

    return transfer_decorator


##############
# Heuristics #
##############

def successor_score(p):
    return p.score


# start
def d_center(p):
    return p.get_maze_d(p.center)


def reach_boundary(p):
    if p.pos == p.center:
        return 1


# hunt and defend
def num_invaders(p):
    return p.num_invaders


def d_remote_invader(p):
    ds = [p.get_maze_d(p.source.most_likely_pos_dict[idx]) for idx in p.invaders_indexes]
    if len(ds) > 0:
        return min(ds)


def d_ally_rev(p):
    if p.in_opponent_territory and p.d_ally is not None:
        return 1.0 / p.d_ally


def stop_punish(p):
    return 1 if p.from_action == Directions.STOP else 0


def reverse_action(p):
    return 1 if p.is_from_reverse_action else 0


# defend

def d_nearby_invader(p):
    return p.d_nearest_invader if len(p.nearby_invaders) > 0 else 0


def danger_dfd(p):
    d = p.d_nearest_nearby_opponent
    if d <= 1 and p.scared_timer > 0:
        return -1
    if d <= 5:
        return 1


# attack
def d_nearest_food(p):
    return p.d_nearest_food if p.food_num > 0 else 0


def collect_food(p):
    v = -p.food_num + 100 * p.score if p.food_num > 0 else 0
    if p.strong:
        v = 100 * v
    return v


def danger_atk(p):
    d = p.d_nearest_nearby_opponent
    if d is None:
        return None

    if d <= 2:
        return 4 / d
    elif d <= 4:
        return 1


def hold_food(p):
    if not p.strong:
        return p.carrying_food_num * p.min_d_opponent_boundary_locations


def store_food(p):
    if p.source is not None:
        return p.returned_food_num - p.source.returned_food_num


def collect_capsule(p):
    if p.capsule_num > 0:
        return -p.capsule_num


def d_capsule_rev(p):
    if p.capsule_num > 0:
        return 1.0 / p.d_nearest_capsule
    return 1.0 / .1


def strong_level(p):
    if p.strong:
        return p.capsule_timer / p.CAPSULE_LAST_TIME


def path_end(p):
    return 1 if len(p.actions) <= 2 else 0


##########
# Agents #
##########


def get_invade_feature_set():
    return FeatureSet([
        (1000, successor_score),
        (-10, d_nearest_food),
        (-1000, danger_atk),
        (5000, collect_food),
        (1000, d_capsule_rev),
        (-1000, stop_punish),
        (-200, path_end),
        (500 * 10000, strong_level),
        (5000, collect_capsule),

        # (-2500, d_ally_rev),
    ])


class StageSM(StateMachine):

    def default(self):
        return get_invade_feature_set, self.invade.__name__

    @state(lambda: FeatureSet([
        (-1, d_center),
        (1000, reach_boundary)
    ]))
    def start(self, player):
        return not player.reach_boundary

    @state(get_invade_feature_set, 1)
    def invade(self, player):
        if player.pos == player.center and not player.reach_boundary:
            player.reach_boundary = True
            return True
        return False

    @state(lambda: FeatureSet([
        (-5000, num_invaders),
        (-10, d_remote_invader),
        (-5000, stop_punish),
        (-5000, reverse_action),
        (-2500, d_ally_rev)
    ]), 2)
    def chase(self, player):
        return player.invader_exists

    @state(lambda: FeatureSet([
        (-10000, num_invaders),
        (-1000, d_nearby_invader),
        (-5000, stop_punish),
        (-100, reverse_action),
        (3000, danger_dfd),
        (-4000, d_ally_rev)
    ]), 3)
    def guard(self, player):
        for o in player.nearby_opponents:
            if player.get_maze_d(o.pos) < 5 and not player.in_opponent_territory:
                return True
        return False


# noinspection PyAttributeOutsideInit,PyAttributeOutsideInit
class MainAgent(CaptureAgent):

    def registerInitialState(self, gameState):
        CaptureAgent.registerInitialState(self, gameState)
        self.player = PlayerState(self, gameState)
        self.stage = StageSM()
        PlayerState.init_beliefs(self.player, gameState)

    def chooseAction(self, gameState):
        p = self.player

        p.update_game_state(gameState)
        p.update_most_likely_opponents_pos()

        self.stage.reset()
        self.stage.act(p)

        fs = self.stage.state

        values = [fs.eval(succ_p) for succ_p in p.succs]

        max_v = max(values)
        best_actions = [a for a, v in zip(p.actions, values) if v == max_v]

        return best_actions[0]
#
