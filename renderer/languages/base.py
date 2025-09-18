# dsl_renderer/base_renderer.py
"""
Base renderer class providing the common interface for all DSL implementations.

This module defines the abstract base class that all DSL renderers must inherit from.
It provides shared mathematical operations and constants, while requiring each
DSL to implement its own evaluation logic.
"""
import abc
import math
import operator
from ..core import AstNode

class BaseRenderer(abc.ABC):
    """
    Abstract base class for DSL renderers.
    
    Provides a common interface and shared functionality for all DSL implementations.
    Each renderer maintains dictionaries of primitives (constants/basic shapes) and
    implementations (functions) that define the DSL's vocabulary.
    
    Subclasses must implement the evaluate() method to define how AST nodes are
    converted into geometric strokes.
    
    Attributes:
        primitives (dict): Constants and basic geometric shapes available in the DSL
        implementations (dict): Function implementations available in the DSL
        
    Examples:
        >>> class MyRenderer(BaseRenderer):
        ...     def evaluate(self, node):
        ...         return self._evaluate_recursive(node)
    """
    def __init__(self):
        self.primitives = {}
        self.implementations = {}
        self._register_shared()

    def _register_shared(self):
        """
        Registers mathematical operations and constants shared across all DSLs.
        
        Sets up basic arithmetic operations (+, -, *, /) and mathematical constants
        (pi) that are available in all DSL implementations. This provides a common
        foundation for mathematical expressions.
        
        Registered primitives:
            - pi: Mathematical constant π
            
        Registered implementations:
            - +, -, *, /: Basic arithmetic operations
        """
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
        Evaluates an AST node to produce a list of geometric strokes.
        
        This is the main method that converts a parsed DSL program (represented
        as an AstNode tree) into a list of strokes that can be rendered as an image.
        Each subclass must implement this method according to its DSL semantics.
        
        Args:
            node: The root AstNode of the program to evaluate
            
        Returns:
            List of strokes. Each stroke can be:
            - numpy array of shape (N, 2) for basic strokes
            - tuple of (stroke_array, color) for colored strokes
            
        Raises:
            NotImplementedError: This is an abstract method that must be overridden
            
        Examples:
            >>> renderer = SomeRenderer()
            >>> ast = parse_program("(C l c)")
            >>> strokes = renderer.evaluate(ast)
            >>> len(strokes) >= 0  # Returns list of strokes
            True
        """
        raise NotImplementedError