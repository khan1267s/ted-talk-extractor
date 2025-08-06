"""
Progress tracking module for TED Talk Processor
Provides real-time progress updates with visual progress bars
"""

import time
import sys
from typing import Optional, Callable
from datetime import datetime, timedelta


class ProgressTracker:
    """Track and display progress for video processing operations"""
    
    def __init__(self, total_steps: int, description: str = "Processing", verbose: bool = True):
        self.total_steps = total_steps
        self.current_step = 0
        self.description = description
        self.verbose = verbose
        self.start_time = None
        self.step_times = []
        
    def start(self):
        """Start tracking progress"""
        self.start_time = time.time()
        if self.verbose:
            print(f"\n{self.description} started at {datetime.now().strftime('%H:%M:%S')}")
            self._draw_progress_bar()
    
    def update(self, step: int = None, message: str = None):
        """Update progress with optional message"""
        if step is not None:
            self.current_step = step
        else:
            self.current_step += 1
            
        step_time = time.time()
        self.step_times.append(step_time)
        
        if self.verbose:
            self._draw_progress_bar(message)
    
    def _draw_progress_bar(self, message: str = None):
        """Draw a visual progress bar in the terminal"""
        if self.total_steps == 0:
            return
            
        progress = self.current_step / self.total_steps
        bar_length = 50
        filled_length = int(bar_length * progress)
        
        # Create progress bar
        bar = '█' * filled_length + '░' * (bar_length - filled_length)
        
        # Calculate time estimates
        elapsed_time = time.time() - self.start_time if self.start_time else 0
        eta = self._calculate_eta(elapsed_time, progress)
        
        # Format output
        output = f"\r{self.description}: [{bar}] {progress*100:.1f}% "
        output += f"({self.current_step}/{self.total_steps}) "
        
        if eta:
            output += f"ETA: {eta} "
            
        if message:
            output += f"- {message}"
            
        # Clear line and print
        sys.stdout.write('\r' + ' ' * 100 + '\r')
        sys.stdout.write(output)
        sys.stdout.flush()
    
    def _calculate_eta(self, elapsed_time: float, progress: float) -> str:
        """Calculate estimated time of arrival"""
        if progress == 0:
            return "calculating..."
            
        total_time = elapsed_time / progress
        remaining_time = total_time - elapsed_time
        
        if remaining_time < 60:
            return f"{int(remaining_time)}s"
        elif remaining_time < 3600:
            return f"{int(remaining_time/60)}m {int(remaining_time%60)}s"
        else:
            hours = int(remaining_time / 3600)
            minutes = int((remaining_time % 3600) / 60)
            return f"{hours}h {minutes}m"
    
    def complete(self, message: str = None):
        """Mark progress as complete"""
        self.current_step = self.total_steps
        
        if self.verbose:
            self._draw_progress_bar(message or "Complete!")
            elapsed = time.time() - self.start_time if self.start_time else 0
            print(f"\n{self.description} completed in {self._format_duration(elapsed)}\n")
    
    def _format_duration(self, seconds: float) -> str:
        """Format duration in human-readable format"""
        if seconds < 60:
            return f"{seconds:.1f} seconds"
        elif seconds < 3600:
            return f"{int(seconds/60)}m {int(seconds%60)}s"
        else:
            hours = int(seconds / 3600)
            minutes = int((seconds % 3600) / 60)
            return f"{hours}h {minutes}m"


class MultiStageProgressTracker:
    """Track progress across multiple stages of processing"""
    
    def __init__(self, stages: list, verbose: bool = True):
        self.stages = stages
        self.verbose = verbose
        self.current_stage_index = 0
        self.stage_trackers = {}
        self.overall_start_time = None
        
    def start(self):
        """Start multi-stage tracking"""
        self.overall_start_time = time.time()
        if self.verbose:
            print(f"\n{'='*60}")
            print(f"Starting multi-stage process with {len(self.stages)} stages:")
            for i, stage in enumerate(self.stages, 1):
                print(f"  {i}. {stage['name']}")
            print(f"{'='*60}\n")
    
    def start_stage(self, stage_name: str, total_steps: int):
        """Start tracking a specific stage"""
        if self.verbose:
            print(f"\n--- Stage {self.current_stage_index + 1}/{len(self.stages)}: {stage_name} ---")
        
        tracker = ProgressTracker(total_steps, stage_name, self.verbose)
        tracker.start()
        self.stage_trackers[stage_name] = tracker
        return tracker
    
    def next_stage(self):
        """Move to the next stage"""
        self.current_stage_index += 1
    
    def complete(self):
        """Complete all stages"""
        if self.verbose:
            total_time = time.time() - self.overall_start_time
            print(f"\n{'='*60}")
            print(f"All stages completed in {self._format_duration(total_time)}")
            print(f"{'='*60}\n")
    
    def _format_duration(self, seconds: float) -> str:
        """Format duration in human-readable format"""
        if seconds < 60:
            return f"{seconds:.1f} seconds"
        elif seconds < 3600:
            return f"{int(seconds/60)}m {int(seconds%60)}s"
        else:
            hours = int(seconds / 3600)
            minutes = int((seconds % 3600) / 60)
            return f"{hours}h {minutes}m"


def track_function_progress(description: str = "Processing", steps: int = None):
    """Decorator to automatically track function progress"""
    def decorator(func: Callable):
        def wrapper(*args, **kwargs):
            # Determine number of steps
            total_steps = steps
            if total_steps is None and args and hasattr(args[0], '__len__'):
                total_steps = len(args[0])
            elif total_steps is None:
                total_steps = 1
                
            tracker = ProgressTracker(total_steps, description)
            tracker.start()
            
            try:
                # Pass tracker to function if it accepts it
                if 'progress_tracker' in func.__code__.co_varnames:
                    kwargs['progress_tracker'] = tracker
                result = func(*args, **kwargs)
                tracker.complete()
                return result
            except Exception as e:
                tracker.complete(f"Failed: {str(e)}")
                raise
                
        return wrapper
    return decorator