import threading

class NavigationState:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(NavigationState, cls).__new__(cls)
                cls._instance.pos_x = None
                cls._instance.pos_y = None
                cls._instance.angle = None
        return cls._instance

    def update_pos(self, x: float, y: float):
        self.pos_x = x
        self.pos_y = y

    def update_angle(self, angle: float):
        self.angle = angle

    def get_pos(self):
        return self.pos_x, self.pos_y

    def get_angle(self):
        return self.angle

nav_state = NavigationState()
