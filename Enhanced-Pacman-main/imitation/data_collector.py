# imitation/enhanced_data_collector.py
import numpy as np
import os
import time
import pickle
import json
from datetime import datetime
import matplotlib.pyplot as plt
import constants
from utils.helpers import pixel_to_grid

class ImprovedDataCollector:
    """Enhanced data collector for recording expert demonstrations for imitation learning.
    
    Features:
    - Records observations, actions, rewards, and positions
    - Visualises the recorded path
    - Analyses action distribution and quality of demonstrations
    - Provides detailed metadata and statistics
    """
    def __init__(self, maze, pellet_grid, save_dir="outputs/imitation_data"):
        """Initialise the data collector.
        
        Args:
            maze: 2D maze grid
            pellet_grid: 2D grid of pellet locations
            save_dir: Directory to save collected data
        """
        self.maze = maze
        self.pellet_grid = pellet_grid
        self.save_dir = save_dir
        self.recordings = []
        self.is_recording = False
        self.current_episode = None
        self.episode_count = 0
        self.total_episodes_saved = 0
        self.action_counts = [0, 0, 0, 0]  # Count of each action type recorded
        
        # Enhanced tracking
        self.path_visualisation = None
        self.path_coverage = {}  # Track visited tiles
        self.last_action_time = None
        self.action_histogram = []  # Track timing of actions
        
        # Create save directory if it doesn't exist
        os.makedirs(save_dir, exist_ok=True)
        
        # Load existing episode count if available
        metadata_file = os.path.join(save_dir, "metadata.json")
        if os.path.exists(metadata_file):
            try:
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                    self.total_episodes_saved = metadata.get('total_episodes', 0)
                    print(f"Found {self.total_episodes_saved} existing episodes in metadata")
            except Exception as e:
                print(f"Error loading metadata: {e}")
        
    def start_recording(self, pacman, active_ghosts):
        """Start recording a new episode."""
        if self.is_recording:
            print("Already recording an episode!")
            return
            
        print("Started recording expert demonstration")
        self.is_recording = True
        
        # Reset path visualisation and coverage
        self.path_visualisation = np.zeros((constants.ROWS, constants.COLS, 3), dtype=np.uint8)
        self.path_coverage = {}
        self.last_action_time = time.time()
        self.action_histogram = []
        
        # Initialise episode data structure
        self.current_episode = {
            'observations': [],
            'actions': [],
            'rewards': [],
            'timestamps': [],
            'positions': [],
            'maze_type': constants.CURRENT_MAZE_TYPE,
            'metadata': {
                'start_time': time.time(),
                'maze_dimensions': (constants.ROWS, constants.COLS),
                'initial_pellet_count': sum(sum(row) for row in self.pellet_grid),
                'version': 2.0  # Version of the data format
            }
        }
        
        # Record initial observation
        initial_obs = pacman._create_observation(active_ghosts, self.pellet_grid, self.maze)
        self._record_step(pacman, active_ghosts, None, 0)  # No action for initial step
        
        print(f"Recording started for episode {self.episode_count + 1}")
        print("Press 'T' to stop recording or 'R' to cancel")
    
    def stop_recording(self, score=0, won=False, cancelled=False):
        """Stop recording the current episode and save the data."""
        if not self.is_recording:
            print("Not currently recording an episode!")
            return
        
        if cancelled:
            print("Recording cancelled - data discarded")
            self.is_recording = False
            self.current_episode = None
            return
            
        # Add episode metadata
        self.current_episode['metadata'].update({
            'end_time': time.time(),
            'duration': time.time() - self.current_episode['metadata']['start_time'],
            'final_score': score,
            'won': won,
            'episode_id': self.episode_count,
            'action_counts': self.action_counts.copy(),
            'total_actions': sum(self.action_counts)
        })
        
        # Calculate action timing statistics
        if self.action_histogram:
            self.current_episode['metadata']['action_timing'] = {
                'min': min(self.action_histogram),
                'max': max(self.action_histogram),
                'mean': sum(self.action_histogram) / len(self.action_histogram),
                'median': sorted(self.action_histogram)[len(self.action_histogram) // 2]
            }
        
        # Add to recordings list
        self.recordings.append(self.current_episode)
        
        # Save this episode
        episode_path = self._save_episode(self.current_episode)
        
        # Generate and save path visualisation
        if len(self.current_episode['positions']) > 0:
            self._save_path_visualisation()
        
        # Generate and display basic statistics
        stats = self._calculate_statistics()
        self._display_statistics(stats)
        
        print(f"Finished recording demonstration. Episode length: {len(self.current_episode['observations'])} steps")
        print(f"Saved to {episode_path}")
        
        self.is_recording = False
        self.current_episode = None
        self.episode_count += 1
        self.total_episodes_saved += 1
        
        # Update metadata file
        self._update_metadata()
    
    def record_step(self, pacman, active_ghosts, action, reward):
        """Record a single step of expert gameplay."""
        if not self.is_recording:
            return
            
        # Calculate action timing
        current_time = time.time()
        if self.last_action_time is not None:
            self.action_histogram.append(current_time - self.last_action_time)
        self.last_action_time = current_time
        
        # Update action counts
        if action is not None and 0 <= action < 4:
            self.action_counts[action] += 1
            
        # Record step
        self._record_step(pacman, active_ghosts, action, reward)
        
        # Update path visualisation
        self._update_path_visualisation(pacman)
    
    def _record_step(self, pacman, active_ghosts, action, reward):
        """Internal method to record a single step."""
        # Get observation using the same method used by the DQN agent
        observation = pacman._create_observation(active_ghosts, self.pellet_grid, self.maze)
        
        # Get current position
        position = pixel_to_grid(pacman.x, pacman.y)
        
        # Add to current episode
        self.current_episode['observations'].append(observation.tolist())
        self.current_episode['actions'].append(action)
        self.current_episode['rewards'].append(reward)
        self.current_episode['timestamps'].append(time.time())
        self.current_episode['positions'].append(position)
        
        # Update path coverage
        pos_key = f"{position[0]},{position[1]}"
        self.path_coverage[pos_key] = self.path_coverage.get(pos_key, 0) + 1
    
    def _update_path_visualisation(self, pacman):
        """Update the path visualisation with current position."""
        if self.path_visualisation is None:
            return
            
        pos = pixel_to_grid(pacman.x, pacman.y)
        if 0 <= pos[1] < constants.ROWS and 0 <= pos[0] < constants.COLS:
            # Use a colour gradient based on time (blue to red)
            progress = len(self.current_episode['observations']) / 1000.0  # Normalise
            progress = min(1.0, progress)  # Cap at 1.0
            
            # Colour: start with blue, transition to green, then red
            if progress < 0.5:
                # Blue to green
                r = int(255 * progress * 2)
                g = int(255 * progress * 2)
                b = int(255 * (1 - progress * 2))
            else:
                # Green to red
                p = (progress - 0.5) * 2
                r = int(255)
                g = int(255 * (1 - p))
                b = 0
                
            self.path_visualisation[pos[1], pos[0]] = [r, g, b]
    
    def _save_path_visualisation(self):
        """Save the path visualisation as an image."""
        if self.path_visualisation is None:
            return
            
        # Create a visualisation with the maze and path
        viz = np.zeros((constants.ROWS, constants.COLS, 3), dtype=np.uint8)
        
        # Add maze walls (blue)
        for row in range(constants.ROWS):
            for col in range(constants.COLS):
                if self.maze[row][col] == 1:
                    viz[row, col] = [0, 0, 100]  # Dark blue for walls
        
        # Add path on top (path visualisation has priority)
        non_zero_mask = self.path_visualisation.sum(axis=2) > 0
        viz[non_zero_mask] = self.path_visualisation[non_zero_mask]
        
        # Upscale image for better visibility (4x)
        scale = 8
        viz_upscaled = np.zeros((constants.ROWS * scale, constants.COLS * scale, 3), dtype=np.uint8)
        for row in range(constants.ROWS):
            for col in range(constants.COLS):
                viz_upscaled[row*scale:(row+1)*scale, col*scale:(col+1)*scale] = viz[row, col]
        
        # Save image
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        viz_path = os.path.join(self.save_dir, f"path_vis_ep{self.episode_count}_{timestamp}.png")
        plt.figure(figsize=(12, 12))
        plt.imshow(viz_upscaled)
        plt.title(f"Path Visualisation - Episode {self.episode_count}")
        plt.axis('off')
        plt.savefig(viz_path)
        plt.close()
        
        print(f"Path visualisation saved to {viz_path}")
    
    def _save_episode(self, episode):
        """Save a single episode to disk."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"episode_{self.episode_count}_{timestamp}.pkl"
        filepath = os.path.join(self.save_dir, filename)
        
        # Save episode data as pickle (for full numeric data)
        with open(filepath, 'wb') as f:
            pickle.dump(episode, f)
        
        # Also save a summary as JSON for easier inspection
        summary = {
            'episode_id': episode['metadata']['episode_id'],
            'maze_type': episode['maze_type'],
            'steps': len(episode['observations']),
            'final_score': episode['metadata']['final_score'],
            'won': episode['metadata']['won'],
            'duration': episode['metadata']['duration'],
            'timestamp': timestamp,
            'action_counts': self.action_counts,
            'coverage_percent': self._calculate_coverage_percent()
        }
        
        summary_filepath = os.path.join(self.save_dir, f"summary_{self.episode_count}_{timestamp}.json")
        with open(summary_filepath, 'w') as f:
            json.dump(summary, f, indent=2)
            
        return filepath
    
    def _update_metadata(self):
        """Update the central metadata file with information about all episodes."""
        metadata_file = os.path.join(self.save_dir, "metadata.json")
        
        metadata = {
            'total_episodes': self.total_episodes_saved,
            'last_updated': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'maze_types': {}
        }
        
        # Scan directory for all episode files and build metadata
        for filename in os.listdir(self.save_dir):
            if filename.startswith("summary_") and filename.endswith(".json"):
                try:
                    with open(os.path.join(self.save_dir, filename), 'r') as f:
                        summary = json.load(f)
                        maze_type = summary.get('maze_type', 'UNKNOWN')
                        
                        if maze_type not in metadata['maze_types']:
                            metadata['maze_types'][maze_type] = {
                                'count': 0,
                                'total_steps': 0,
                                'wins': 0
                            }
                            
                        metadata['maze_types'][maze_type]['count'] += 1
                        metadata['maze_types'][maze_type]['total_steps'] += summary.get('steps', 0)
                        metadata['maze_types'][maze_type]['wins'] += 1 if summary.get('won', False) else 0
                except Exception as e:
                    print(f"Error processing summary file {filename}: {e}")
        
        # Save updated metadata
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def _calculate_coverage_percent(self):
        """Calculate the percentage of navigable maze covered by the demonstration."""
        navigable_cells = sum(row.count(0) for row in self.maze)
        visited_cells = len(self.path_coverage)
        
        return (visited_cells / navigable_cells) * 100 if navigable_cells > 0 else 0
    
    def _calculate_statistics(self):
        """Calculate statistics for the recorded episode."""
        if not self.current_episode:
            return {}
            
        stats = {
            'steps': len(self.current_episode['observations']),
            'duration': self.current_episode['metadata']['duration'],
            'score': self.current_episode['metadata']['final_score'],
            'won': self.current_episode['metadata']['won'],
            'action_counts': self.action_counts.copy(),
            'coverage_percent': self._calculate_coverage_percent(),
            'revisit_rate': self._calculate_revisit_rate()
        }
        
        return stats
    
    def _calculate_revisit_rate(self):
        """Calculate how often positions were revisited."""
        if not self.path_coverage:
            return 0.0
            
        visit_counts = list(self.path_coverage.values())
        avg_visits = sum(visit_counts) / len(visit_counts)
        
        # 1.0 means every cell visited exactly once, higher is more revisits
        return avg_visits
    
    def _display_statistics(self, stats):
        """Display statistics about the recorded episode."""
        if not stats:
            return
            
        print("\n=== Demonstration Statistics ===")
        print(f"Steps: {stats['steps']}")
        print(f"Duration: {stats['duration']:.2f} seconds")
        print(f"Score: {stats['score']}")
        print(f"Won: {stats['won']}")
        
        # Action distribution
        action_names = ["Right", "Left", "Down", "Up"]
        total_actions = sum(stats['action_counts'])
        print("\nAction Distribution:")
        if total_actions > 0:
            for i, count in enumerate(stats['action_counts']):
                print(f"  {action_names[i]}: {count} ({count/total_actions*100:.1f}%)")
        
        print(f"\nMaze Coverage: {stats['coverage_percent']:.1f}%")
        print(f"Revisit Rate: {stats['revisit_rate']:.2f}")
        
        # Evaluate demonstration quality
        quality_score = self._evaluate_demonstration_quality(stats)
        quality_rating = "Excellent" if quality_score > 0.8 else "Good" if quality_score > 0.6 else "Average" if quality_score > 0.4 else "Poor"
        print(f"\nDemonstration Quality: {quality_rating} ({quality_score:.2f}/1.0)")
        print("==============================")
    
    def _evaluate_demonstration_quality(self, stats):
        """Evaluate the quality of the demonstration for training purposes."""
        quality = 0.0
        
        # Factor 1: Coverage (30%)
        coverage_score = min(1.0, stats['coverage_percent'] / 70)  # 70% coverage is excellent
        quality += 0.3 * coverage_score
        
        # Factor 2: Action balance (20%)
        if sum(stats['action_counts']) > 0:
            action_balance = min(stats['action_counts']) / max(stats['action_counts'])
            action_score = min(1.0, action_balance * 2)  # 1:2 ratio is considered good
            quality += 0.2 * action_score
        
        # Factor 3: Won the game (20%)
        quality += 0.2 if stats['won'] else 0.0
        
        # Factor 4: Appropriate length (15%)
        # A good demonstration should have enough steps, but not too many (revisits)
        length_score = 0.0
        if 100 <= stats['steps'] <= 500:
            length_score = 1.0
        elif 50 <= stats['steps'] < 100 or 500 < stats['steps'] <= 1000:
            length_score = 0.7
        elif stats['steps'] > 1000:
            length_score = 0.3
        else:
            length_score = 0.1
        quality += 0.15 * length_score
        
        # Factor 5: Revisit rate (15%) - lower is better, up to a point
        revisit_score = 0.0
        if 1.0 <= stats['revisit_rate'] <= 2.0:
            revisit_score = 1.0
        elif 2.0 < stats['revisit_rate'] <= 3.0:
            revisit_score = 0.7
        elif 3.0 < stats['revisit_rate'] <= 5.0:
            revisit_score = 0.4
        else:
            revisit_score = 0.2
        quality += 0.15 * revisit_score
        
        return quality
    
    def save_all_recordings(self):
        """Save all recorded episodes to a single file."""
        if not self.recordings:
            print("No recordings to save!")
            return
            
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"all_episodes_{timestamp}.pkl"
        filepath = os.path.join(self.save_dir, filename)
        
        with open(filepath, 'wb') as f:
            pickle.dump(self.recordings, f)
            
        print(f"Saved all {len(self.recordings)} episodes to {filepath}")
        
    def analyse_all_demonstrations(self):
        """Analyse all demonstrations in the save directory and generate summary report."""
        summaries = []
        
        # Collect all summary files
        for filename in os.listdir(self.save_dir):
            if filename.startswith("summary_") and filename.endswith(".json"):
                try:
                    with open(os.path.join(self.save_dir, filename), 'r') as f:
                        summary = json.load(f)
                        summaries.append(summary)
                except Exception as e:
                    print(f"Error processing summary file {filename}: {e}")
        
        if not summaries:
            print("No demonstration summaries found!")
            return
        
        print(f"\n=== Analysis of {len(summaries)} Demonstrations ===")
        
        # Basic statistics
        total_steps = sum(s.get('steps', 0) for s in summaries)
        total_wins = sum(1 for s in summaries if s.get('won', False))
        avg_steps = total_steps / len(summaries) if summaries else 0
        
        print(f"Total demonstrations: {len(summaries)}")
        print(f"Total steps recorded: {total_steps}")
        print(f"Average steps per demonstration: {avg_steps:.1f}")
        print(f"Win rate: {total_wins}/{len(summaries)} ({total_wins/len(summaries)*100:.1f}%)")
        
        # Action distribution across all demonstrations
        action_totals = [0, 0, 0, 0]
        for s in summaries:
            counts = s.get('action_counts', [0, 0, 0, 0])
            for i in range(4):
                action_totals[i] += counts[i] if i < len(counts) else 0
        
        print("\nOverall Action Distribution:")
        action_names = ["Right", "Left", "Down", "Up"]
        total_actions = sum(action_totals)
        if total_actions > 0:
            for i, count in enumerate(action_totals):
                print(f"  {action_names[i]}: {count} ({count/total_actions*100:.1f}%)")
        
        # Coverage analysis
        coverage_values = [s.get('coverage_percent', 0) for s in summaries]
        avg_coverage = sum(coverage_values) / len(coverage_values) if coverage_values else 0
        
        print(f"\nAverage maze coverage: {avg_coverage:.1f}%")
        print(f"Best coverage: {max(coverage_values):.1f}%")
        
        # Generate and save report
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = os.path.join(self.save_dir, f"demonstration_analysis_{timestamp}.txt")
        
        with open(report_path, 'w') as f:
            f.write(f"=== Demonstration Analysis ===\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(f"Total demonstrations: {len(summaries)}\n")
            f.write(f"Total steps recorded: {total_steps}\n")
            f.write(f"Average steps per demonstration: {avg_steps:.1f}\n")
            f.write(f"Win rate: {total_wins}/{len(summaries)} ({total_wins/len(summaries)*100:.1f}%)\n\n")
            
            f.write("Overall Action Distribution:\n")
            for i, count in enumerate(action_totals):
                f.write(f"  {action_names[i]}: {count} ({count/total_actions*100:.1f}%)\n")
            
            f.write(f"\nAverage maze coverage: {avg_coverage:.1f}%\n")
            f.write(f"Best coverage: {max(coverage_values):.1f}%\n\n")
            
            f.write("Individual Demonstration Summaries:\n")
            for i, s in enumerate(summaries):
                f.write(f"\nDemo #{i+1}:\n")
                f.write(f"  Steps: {s.get('steps', 0)}\n")
                f.write(f"  Won: {s.get('won', False)}\n")
                f.write(f"  Coverage: {s.get('coverage_percent', 0):.1f}%\n")
                
                # Action distribution for this demo
                counts = s.get('action_counts', [0, 0, 0, 0])
                demo_total = sum(counts)
                if demo_total > 0:
                    f.write("  Actions:\n")
                    for j, count in enumerate(counts):
                        if j < len(action_names):
                            f.write(f"    {action_names[j]}: {count} ({count/demo_total*100:.1f}%)\n")
        
        print(f"\nDetailed analysis saved to {report_path}")
        print("===============================\n")