# dsl_renderer/probabilistic.py
import math
import numpy as np
from typing import Dict, Union
from ..core import AstNode
from .base import BaseRenderer


## --- Noise & Drawing Hyperparameters (Updated to match legacy.py) ---
NOISE_PARAMS = {
    'sigma': 0.02, 'smooth_amp': 0.04, 'base_freq': 0.5,
    'octaves': 3, 'persistence': 0.5,
}
BEZIER_NUM_POINTS = 40
DOT_NUM_POINTS = 20
State = Dict[str, Union[np.ndarray, float]]


## --- Helper functions for curve generation ---
def _generate_smooth_noise(num_points, amp, freq, oct, pers):
    # return array of zeros if amplitude or frequency is 0
    if amp == 0 or freq == 0: 
        return np.zeros((num_points, 2))  
    # generate noise for x and y coordinates using the given parameters
    t = np.linspace(0, 1, num_points)
    noise_x, noise_y = np.zeros(num_points), np.zeros(num_points)
    for _ in range(int(oct)):
        phase_x, phase_y = np.random.uniform(0, 2 * math.pi, 2)
        noise_x += amp * np.sin(t * freq * 2 * math.pi + phase_x)
        noise_y += amp * np.sin(t * freq * 2 * math.pi + phase_y)
        amp *= pers
        freq *= 2
    envelope = np.sin(t * math.pi)
    return np.column_stack((noise_x * envelope, noise_y * envelope))


def _bezier(p0, p1, p2, p3, num_points, smooth_noise):
    t = np.linspace(0, 1, num_points)[:, np.newaxis]
    points = ((1-t)**3*p0 + 3*(1-t)**2*t*p1 + 3*(1-t)*t**2*p2 + t**3*p3)
    return points + smooth_noise


## --- Stateful Drawing Functions ---
def _curve(d_angle, length, bend, state):
    n = NOISE_PARAMS
    d_angle_n = d_angle + np.random.normal(0, n['sigma'])
    length_n = abs(length + np.random.normal(0, n['sigma'] * length))
    bend_n = bend + np.random.normal(0, n['sigma'])
    
    start_pos, start_angle = state['pos'], state['angle']
    end_angle = start_angle + d_angle_n
    avg_angle = start_angle + d_angle_n / 2.0
    end_pos = start_pos + np.array([length_n * math.cos(avg_angle), length_n * math.sin(avg_angle)])
    
    ctrl_len = bend_n * length_n
    p1 = start_pos + ctrl_len * np.array([math.cos(start_angle), math.sin(start_angle)])
    p2 = end_pos - ctrl_len * np.array([math.cos(end_angle), math.sin(end_angle)])
    
    noise = _generate_smooth_noise(BEZIER_NUM_POINTS, n['smooth_amp'], n['base_freq'], n['octaves'], n['persistence'])
    stroke = _bezier(start_pos, p1, p2, end_pos, BEZIER_NUM_POINTS, noise)
    return [stroke], {'pos': end_pos, 'angle': end_angle}


def _dot(radius, state):
    n_sigma = NOISE_PARAMS['sigma']
    radius_n = abs(radius + np.random.normal(0, n_sigma * radius * 0.5))
    center_n = state['pos'] + np.random.normal(0, n_sigma * 0.2, 2)
    thetas = np.linspace(0, 2 * math.pi, DOT_NUM_POINTS)
    dot_stroke = center_n + radius_n * np.array([np.cos(thetas), np.sin(thetas)]).T
    return [dot_stroke], state


def _turn(new_angle, state):
    return [], {'pos': state['pos'], 'angle': float(new_angle)}


class ProbabilisticRenderer(BaseRenderer):
    """Renderer for the stateful, probabilistic drawing DSL."""
    def __init__(self):
        super().__init__()
        self.drawing_implementations = {}
        self._register_dsl_specific()
        
    def _register_dsl_specific(self):
        self.drawing_implementations.update({
            'curve': _curve, 'dot': _dot, 'turn': _turn,
        })

    def evaluate(self, node: AstNode):
        initial_state = {'pos': np.array([0.0, 0.0]), 'angle': -math.pi / 2}
        strokes, _ = self._evaluate_recursive(node, initial_state)
        return strokes

    def _evaluate_recursive(self, node, state):
        if not isinstance(node, AstNode):
            return node, state
            
        # check if the node is a 'C' or 'lift' node and handle them accordingly
        if node.name == 'C':
            all_strokes, current_state = [], state
            for arg in node.args:
                strokes, current_state = self._evaluate_recursive(arg, current_state)
                all_strokes.extend(strokes)
            return all_strokes, current_state
        if node.name == 'lift':
            args = [self._evaluate_recursive(arg, state)[0] for arg in node.args]
            angle = state['angle'] if len(args) == 2 else float(args[2])
            new_state = {'pos': np.array([float(args[0]), float(args[1])]), 'angle': angle}
            return [], new_state
            
        # evaluate arguments for the default case
        evaluated_args = [self._evaluate_recursive(arg, state)[0] for arg in node.args]
        
        # check for stateful drawing functions and pass the state
        if node.name in self.drawing_implementations:
            return self.drawing_implementations[node.name](*evaluated_args, state=state)

        # check for stateless math functions (from BaseRenderer) and do NOT pass the state
        if node.name in self.implementations:
            result = self.implementations[node.name](*evaluated_args)
            return result, state

        if node.name in self.primitives:
            return self.primitives[node.name], state
            
        raise ValueError(f"Unknown function or primitive: {node.name}")