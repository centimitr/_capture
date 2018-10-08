import operator

from game import AgentState
from game import Directions
from util import nearestPoint
from _world_state import WorldState
from captureAgents import CaptureAgent
import util


class PlayerState:

    @staticmethod
    def extend_agent_state(instance_or_class):
        @property
        def direction(self):
            return self.getDirection()

        @property
        def pos(self):
            return self.getPosition()

        instance_or_class.dir = direction
        instance_or_class.pos = pos

    beliefs = None

    @staticmethod
    def init_beliefs(player, gameState):
        PlayerState.beliefs = [util.Counter()] * gameState.getNumAgents()

        # initial positions
        for i, val in enumerate(PlayerState.beliefs):
            if i in player.opponents_indexes:
                PlayerState.beliefs[i][gameState.getInitialAgentPosition(i)] = 1.0

    @staticmethod
    def observe_opponent(player, idx, noisy_d):
        prob = util.Counter()
        for pos in player.legal_locations:
            prob[pos] += player.gameState.getDistanceProb(player.get_manhattan_d(pos), noisy_d)

        for pos in player.legal_locations:
            PlayerState.beliefs[idx][pos] *= prob[pos]

        # print noisyDistance

    @staticmethod
    def observe_opponents(player):
        noisy_ds = player.gameState.getAgentDistances()
        for idx in player.opponents_indexes:
            PlayerState.observe_opponent(player, idx, noisy_ds[idx])

    @staticmethod
    def elapse_time(player):
        for agent, old_belief in enumerate(PlayerState.beliefs):
            if agent in player.opponents_indexes:
                beliefs = util.Counter()

                pos = player.gameState.getAgentPosition(agent)
                # real positions
                if pos is not None:
                    beliefs[pos] = 1.0
                else:
                    # no real positions, infer by beliefs
                    for pos in old_belief:
                        if pos in player.legal_locations and old_belief[pos] > 0:
                            possible_locations = PlayerState.get_possible_locations(player, pos)
                            for x, y in possible_locations:
                                beliefs[x, y] += old_belief[pos] * possible_locations[x, y]
                    if len(beliefs) == 0:
                        prev_state = player._agent.getPreviousObservation()
                        if prev_state is not None and prev_state.getAgentPosition(agent) is not None:
                            beliefs[prev_state.getInitialAgentPosition(agent)] = 1.0
                        else:
                            for p in player.legal_locations:
                                beliefs[p] = 1.0
                PlayerState.beliefs[agent] = beliefs

    @staticmethod
    def get_possible_locations(player, pos):
        distribution = util.Counter()

        x, y = pos
        possible_actions = [
            (x, y),
            (x + 1, y),
            (x - 1, y),
            (x, y + 1),
            (x, y - 1),
        ]
        for a in possible_actions:
            if a in player.legal_locations:
                distribution[a] = 1

        return distribution

    def __init__(self, captureAgent, gameState, source_player_state=None, from_action=None):
        self.CAPSULE_LAST_TIME = 40 * 3

        self._agent = captureAgent
        self.index = self._agent.index
        self.gameState = gameState
        self.world = WorldState(gameState)
        self.source = source_player_state

        self.from_action = from_action
        self.capsule_timer = 0

        self.most_likely_pos_dict = [None] * 4
        self.reach_boundary = False
        self.center = None
        self.legal_locations = [p for p in gameState.getWalls().asList(False) if p[1] > 1]

        self.__init_set_center()

    def __init_set_center(self):
        self.reach_boundary = False

        x = self.world.map_width / 2
        y = self.world.map_height / 2
        if self.red:
            x = x - 1

        locations = []
        if self.senior:
            for i in range(self.world.map_height - y):
                if not self.world.has_wall(x, y):
                    locations.append((x, y))
                y = y + 1
        else:
            for i in range(y):
                if not self.world.has_wall(x, y):
                    locations.append((x, y))
                y = y - 1

        min_d = float('inf')
        min_pos = None
        for pos in locations:
            d = self.get_maze_d(pos)
            if d <= min_d:
                min_d = d
                min_pos = pos
        self.center = min_pos

    def update_game_state(self, gameState):
        # self.extend_game_state(gameState)
        self.gameState = gameState
        self.world = WorldState(gameState)

        # when got a capsule
        if self.pos in self.capsules:
            print 'got capsule'
            self.capsule_timer = self.CAPSULE_LAST_TIME

        # update capsule time left
        if self.strong:
            self.capsule_timer -= 1

    # update most likely opponents pos
    def update_most_likely_opponents_pos(self):
        PlayerState.observe_opponents(self)

        for idx in self.opponents_indexes:
            PlayerState.beliefs[idx].normalize()
            self.most_likely_pos_dict[idx] = max(PlayerState.beliefs[idx].iteritems(), key=operator.itemgetter(1))[0]

        # Do next time step
        PlayerState.elapse_time(self)

    ##########
    # agents #
    ##########

    @property
    def agent(self):
        return self.gameState.getAgentState(self.index)

    @property
    def ally_index(self):
        return next(idx for idx in self._agent.getTeam(self.gameState) if idx != self.index)

    @property
    def ally(self):
        return self.gameState.getAgentState(self.ally_index)

    @property
    def opponents_indexes(self):
        return self._agent.getOpponents(self.gameState)

    @property
    def opponents(self):
        return [self.gameState.getAgentState(idx) for idx in self.opponents_indexes]

    def get_opponent_index(self, opponent):
        return next(idx for (idx, o) in zip(self.opponents_indexes, self.opponents) if o == opponent)

    @property
    def nearby_opponents(self):
        return [o for o in self.opponents if o.pos is not None]

    @property
    def invaders(self):
        return [o for o in self.opponents if o.isPacman]

    @property
    def num_invaders(self):
        return len(self.invaders)

    @property
    def invaders_indexes(self):
        return [idx for (o, idx) in zip(self.opponents, self.opponents_indexes) if o.isPacman]

    @property
    def nearby_invaders(self):
        return [o for o in self.invaders if o.pos is not None]

    @property
    def invader_exists(self):
        return len(self.invaders) > 0

    ###########
    # actions #
    ###########

    @property
    def former_action(self):
        return self.gameState.getAgentState(self.index).configuration.direction

    @property
    def is_from_reverse_action(self):
        former_action = self.source.gameState.getAgentState(self.index).configuration.direction
        return self.from_action == Directions.REVERSE[former_action]

    ##############
    # attributes #
    ##############

    @property
    def dir(self):
        return self.agent.dir

    @property
    def pos(self):
        return self.agent.pos

    @property
    def pos_x(self):
        return self.agent.pos[0]

    @property
    def pos_y(self):
        return self.agent.pos[1]

    @property
    def score(self):
        return self._agent.getScore(self.gameState)

    @property
    def red(self):
        return self._agent.red

    @property
    def strong(self):
        return self.capsule_timer > 0

    @property
    def senior(self):
        return self.index < self.ally_index

    ###########
    # objects #
    ###########

    @property
    def defending_foods(self):
        return self._agent.getFoodYouAreDefending(self.gameState).asList()

    @property
    def defending_food_num(self):
        return len(self.defending_foods)

    @property
    def foods(self):
        return self._agent.getFood(self.gameState).asList()

    @property
    def food_num(self):
        return len(self.foods)

    @property
    def carrying_food_num(self):
        return self.agent.numCarrying

    @property
    def returned_food_num(self):
        return self.agent.numReturned

    @property
    def scared_timer(self):
        return self.agent.scaredTimer

    @property
    def boundary_x(self):
        return self.world.get_boundary_x(self._agent.red)

    @property
    def capsules(self):
        return self._agent.getCapsules(self.gameState)

    @property
    def capsule_num(self):
        return len(self.capsules)

    @property
    def at_boundary(self):
        return self.pos_x == self.boundary_x

    @property
    def boundary_locations(self):
        return self.world.get_boundary_locations(self._agent.red)

    @property
    def opponent_boundary_locations(self):
        return self.world.get_boundary_locations(not self._agent.red)

    @property
    def min_d_boundary_locations(self):
        return min([self.get_maze_d(pos) for pos in self.boundary_locations])

    @property
    def min_d_opponent_boundary_locations(self):
        return min([self.get_maze_d(pos) for pos in self.opponent_boundary_locations])

    @property
    def d_nearest_food(self):
        return min([self.get_maze_d(food_pos) for food_pos in self.foods]) if len(self.foods) > 0 else None

    @property
    def d_nearest_nearby_opponent(self):
        return min([self.get_maze_d(o.pos) for o in self.nearby_opponents]) if len(self.nearby_opponents) > 0 else None

    @property
    def d_nearest_invader(self):
        return min([self.get_maze_d(invader.pos) for invader in self.nearby_invaders])

    @property
    def d_nearest_capsule(self):
        return min([self.get_maze_d(c_pos) for c_pos in self.capsules]) if len(self.capsules) > 0 else None

    @property
    def d_ally(self):
        if self.senior:
            return None
        d = self.get_maze_d(self.ally.pos)
        return 0.5 if d == 0 else d

    @property
    def in_opponent_territory(self):
        return self.agent.isPacman

    def get_maze_d(self, pos):
        return self._agent.getMazeDistance(self.pos, pos)

    def get_manhattan_d(self, pos):
        x, y = pos
        return abs(self.pos_x - x) + abs(self.pos_y - y)

    def get_invader_pos(self, invader):
        return self.most_likely_pos_dict[self.get_opponent_index(invader)]

    ##########
    # future #
    ##########

    @property
    def actions(self):
        return self.gameState.getLegalActions(self.index)

    def _get_succ(self, gameState, action):
        succ = gameState.generateSuccessor(self.index, action)
        pos = succ.getAgentState(self.index).getPosition()
        if pos != nearestPoint(pos):
            return succ.generateSuccessor(self.index, action)
        return succ

    @property
    def succs(self):
        return [PlayerState(self._agent, self._get_succ(self.gameState, a), source_player_state=self, from_action=a) for
                a in
                self.actions]
