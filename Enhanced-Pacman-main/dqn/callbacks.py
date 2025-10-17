from stable_baselines3.common.callbacks import BaseCallback
import numpy as np

class TensorboardScoreCallback(BaseCallback):
    """Custom callback for logging score and completion rate to TensorBoard."""
    def __init__(self, verbose=0):
        super(TensorboardScoreCallback, self).__init__(verbose)
        self.episode_scores = []
        self.episode_completion_rates = []
    
    def _on_step(self):
        if len(self.model.ep_info_buffer) > 0 and self.locals.get("dones")[0]:
            latest_info = self.model.ep_info_buffer[-1]
            
            if 'score' in latest_info:
                score = latest_info.get('score', 0)
                self.episode_scores.append(score)
                self.logger.record('pacman/score', score)
                
                if len(self.episode_scores) >= 10:
                    avg_score = np.mean(self.episode_scores[-10:])
                    self.logger.record('pacman/avg_score_10', avg_score)
            
            if 'completion_rate' in latest_info:
                completion_rate = latest_info.get('completion_rate', 0)
                self.episode_completion_rates.append(completion_rate)
                self.logger.record('pacman/completion_rate', completion_rate)
                
                if len(self.episode_completion_rates) >= 10:
                    avg_completion = np.mean(self.episode_completion_rates[-10:])
                    self.logger.record('pacman/avg_completion_10', avg_completion)
        
        return True