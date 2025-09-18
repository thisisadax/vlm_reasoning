# dsl_renderer/base_renderer.py
import abc
import math
import operator
from ..core import AstNode

class BaseRenderer(abc.ABC):
    """Abstract base class for a DSL renderer."""
    def __init__(self):
        self.primitives = {}
        self.implementations = {}
        self._register_shared()

    def _register_shared(self):
        """Registers primitives and functions shared across all DSLs."""
        # Shared constants
        self.primitives["pi"] = math.pi
        
        # Shared arithmetic operations
        self.implementations.update({
            "/": operator.truediv,
            "*": operator.mul,
            "-": operator.sub,
            "+": operator.add,
        })

    @abc.abstractmethod
    def evaluate(self, node: AstNode):
        """
        Evaluates an AST node to produce a list of strokes.
        This method must be implemented by subclasses.
        """
        raise NotImplementedError