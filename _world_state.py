class WorldState:

    def __init__(self, gameState):
        self.gameState = gameState

    def update_game_state(self, gameState):
        self.gameState = gameState

    @property
    def walls(self):
        return self.gameState.getWalls().asList()

    def has_wall(self, x, y):
        return self.gameState.hasWall(x, y)

    @property
    def map_size(self):
        return self.gameState.getWalls().width, self.gameState.getWalls().height

    @property
    def map_height(self):
        return self.gameState.getWalls().height

    @property
    def map_width(self):
        return self.gameState.getWalls().width

    @property
    def locations(self):
        return [p for p in self.gameState.getWalls().asList(False) if p[1] > 1]

    def _locations_in_col(self, col_x):
        return [(x, y) for (x, y) in self.locations if x == col_x]

    def get_boundary_x(self, is_red):
        w, _ = self.map_size
        return w / 2 - 1 if is_red else w / 2

    def get_boundary_locations(self, is_red):
        return self._locations_in_col(self.get_boundary_x(is_red))
