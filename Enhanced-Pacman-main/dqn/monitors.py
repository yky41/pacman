from stable_baselines3.common.monitor import Monitor
class PacmanMonitor(Monitor):
    """Custom Monitor wrapper that tracks scores and completion rate."""
    
    def __init__(self, env, filename=None, info_keywords=('score', 'pellets_remaining')):
        super().__init__(env, filename, info_keywords=info_keywords)
        observation, _ = env.reset()
        self.initial_pellets = sum(sum(row) for row in env.pellet_grid)
        env.reset()
    
    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        
        if terminated or truncated:
            pellets_remaining = info.get('pellets_remaining', 0)
            pellets_eaten = self.initial_pellets - pellets_remaining
            completion_rate = pellets_eaten / self.initial_pellets if self.initial_pellets > 0 else 0
            info['completion_rate'] = completion_rate
            
        return super().step(action)