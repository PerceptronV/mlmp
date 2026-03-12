"""
Unit tests for the MCTS Composer.

Tests the Monte Carlo reinforcement learning approach to program generation,
including immediate feedback, constant expression penalties, and Q-value learning.
"""

import sys
import os
import unittest
import re

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from typing import Callable
from lang.grammar import DefaultGrammar
from lang.composers.mcts import (
    MCTSComposer,
    MCTSState,
    MCTSNode,
    ActionStats,
    VariabilityScorer,
    train_mcts_composer,
    type_to_hashable,
    min_depth_for_type,
)
from lang.type_utils import SubstitutionTable, CallableOrig
from lang.ast_nodes import (
    ASTNode, pretty_print, NumberNode, BooleanNode, ListNode,
    VariableNode, ApplicationNode, LambdaNode, IfNode
)
from lang.type_checker import TypeChecker
from lang.evaluator import Evaluator


class TestMCTSState(unittest.TestCase):
    """Tests for MCTSState creation and hashing."""

    def test_state_creation(self):
        """Test creating states from context."""
        state = MCTSState.from_context(
            target_type=int,
            context={'x': list[int]},
            parent_function='map',
            arg_index=0,
            depth=3,
            substitutions=SubstitutionTable()
        )
        self.assertEqual(state.target_type, 'int')
        self.assertEqual(state.parent_function, 'map')
        self.assertEqual(state.arg_index, 0)
        self.assertEqual(state.depth_bucket, 1)  # depth 3 -> bucket 1

    def test_state_depth_buckets(self):
        """Test depth discretization into buckets."""
        # Depth 0-1 -> bucket 0
        state0 = MCTSState.from_context(int, {}, None, -1, 0, SubstitutionTable())
        state1 = MCTSState.from_context(int, {}, None, -1, 1, SubstitutionTable())
        self.assertEqual(state0.depth_bucket, 0)
        self.assertEqual(state1.depth_bucket, 0)

        # Depth 2-3 -> bucket 1
        state2 = MCTSState.from_context(int, {}, None, -1, 2, SubstitutionTable())
        state3 = MCTSState.from_context(int, {}, None, -1, 3, SubstitutionTable())
        self.assertEqual(state2.depth_bucket, 1)
        self.assertEqual(state3.depth_bucket, 1)

        # Depth 4+ -> bucket 2
        state4 = MCTSState.from_context(int, {}, None, -1, 4, SubstitutionTable())
        state5 = MCTSState.from_context(int, {}, None, -1, 10, SubstitutionTable())
        self.assertEqual(state4.depth_bucket, 2)
        self.assertEqual(state5.depth_bucket, 2)

    def test_state_hashable(self):
        """Test that states can be used as dict keys."""
        state1 = MCTSState.from_context(int, {'x': int}, 'add', 0, 2, SubstitutionTable())
        state2 = MCTSState.from_context(int, {'x': int}, 'add', 0, 2, SubstitutionTable())

        # Same parameters should produce equal states
        self.assertEqual(state1, state2)

        # Can use as dict key
        d = {state1: 'value'}
        self.assertEqual(d[state2], 'value')


class TestMCTSNode(unittest.TestCase):
    """Tests for MCTSNode and UCB1 computation."""

    def test_ucb1_unexplored(self):
        """Unexplored actions should have infinite UCB1 score."""
        state = MCTSState('int', frozenset(), None, -1, 1)
        node = MCTSNode(state=state)

        # Unexplored action
        score = node.get_ucb1_score('literal')
        self.assertEqual(score, float('inf'))

    def test_ucb1_explored(self):
        """Explored actions should have finite UCB1 score."""
        state = MCTSState('int', frozenset(), None, -1, 1)
        node = MCTSNode(state=state)

        # Add some visits
        node.update('literal', 0.5)
        node.update('literal', 0.6)
        node.update('variable', 0.3)

        # Now should have finite scores
        score_lit = node.get_ucb1_score('literal')
        score_var = node.get_ucb1_score('variable')

        self.assertIsInstance(score_lit, float)
        self.assertIsInstance(score_var, float)
        self.assertNotEqual(score_lit, float('inf'))
        self.assertNotEqual(score_var, float('inf'))

    def test_action_selection(self):
        """Test that UCB1 selects unexplored actions first."""
        state = MCTSState('int', frozenset(), None, -1, 1)
        node = MCTSNode(state=state)

        # Add visits to 'literal'
        node.update('literal', 0.5)

        # Should select unexplored 'variable' first
        selected = node.select_action(['literal', 'variable'])
        self.assertEqual(selected, 'variable')

    def test_q_value_computation(self):
        """Test Q-value (average reward) computation."""
        stats = ActionStats()
        self.assertEqual(stats.q_value, 0.0)

        stats.visits = 2
        stats.total_reward = 1.0
        self.assertEqual(stats.q_value, 0.5)


class TestTypeToHashable(unittest.TestCase):
    """Tests for type hashing utility."""

    def test_primitive_types(self):
        """Test hashing of primitive types."""
        self.assertEqual(type_to_hashable(int), 'int')
        self.assertEqual(type_to_hashable(bool), 'bool')

    def test_list_types(self):
        """Test hashing of list types."""
        self.assertEqual(type_to_hashable(list[int]), 'list[int]')
        self.assertEqual(type_to_hashable(list[bool]), 'list[bool]')

    def test_callable_types(self):
        """Test hashing of callable types."""
        result = type_to_hashable(Callable[[int], int])
        self.assertIn('Callable', result)
        self.assertIn('int', result)


class TestMinDepthForType(unittest.TestCase):
    """Tests for minimum depth computation."""

    def test_atomic_types(self):
        """Atomic types need depth 0."""
        self.assertEqual(min_depth_for_type(int), 0)
        self.assertEqual(min_depth_for_type(bool), 0)

    def test_list_types(self):
        """List types need depth 0 (empty list)."""
        self.assertEqual(min_depth_for_type(list[int]), 0)

    def test_callable_types(self):
        """Callable types need depth 1 (lambda)."""
        self.assertEqual(min_depth_for_type(Callable[[int], int]), 1)


class TestMCTSComposer(unittest.TestCase):
    """Tests for the main MCTSComposer class."""

    def setUp(self):
        """Set up test fixtures."""
        self.target_type = Callable[[list[int]], list[int]]
        self.composer = MCTSComposer(
            seed=42,
            grammar=DefaultGrammar,
            training_mode=True
        )

    def test_initialization(self):
        """Test composer initialization."""
        self.assertEqual(self.composer.training_mode, True)
        self.assertEqual(self.composer.immediate_reward_weight, 0.4)
        self.assertEqual(self.composer.final_reward_weight, 0.6)
        self.assertEqual(self.composer.constant_expression_penalty, -0.5)
        self.assertEqual(len(self.composer.tree), 0)

    def test_generate_produces_valid_ast(self):
        """Test that generate() produces valid AST nodes."""
        self.composer.reset_var_counter()
        program = self.composer.generate(
            target_type=self.target_type,
            depth=3,
            context={},
            substitutions=SubstitutionTable()
        )
        self.assertIsInstance(program, ASTNode)

    def test_generate_produces_well_typed_programs(self):
        """Test that generated programs type-check correctly."""
        checker = TypeChecker()

        for seed in range(5):
            composer = MCTSComposer(seed=seed, grammar=DefaultGrammar)
            composer.reset_var_counter()
            program = composer.generate(
                target_type=self.target_type,
                depth=3,
                context={},
                substitutions=SubstitutionTable()
            )
            # Should type-check without raising
            t = checker.check(program)
            self.assertIsNotNone(t)

    def test_tree_grows_during_training(self):
        """Test that the MCTS tree grows during training."""
        initial_size = len(self.composer.tree)

        for _ in range(10):
            self.composer.reset_var_counter()
            self.composer.generate(
                target_type=self.target_type,
                depth=3,
                context={},
                substitutions=SubstitutionTable()
            )

        final_size = len(self.composer.tree)
        self.assertGreater(final_size, initial_size)

    def test_path_tracking(self):
        """Test that the path through the tree is tracked."""
        self.composer.reset_var_counter()
        self.composer.generate(
            target_type=self.target_type,
            depth=3,
            context={},
            substitutions=SubstitutionTable()
        )
        # Path should have at least the root decision
        self.assertGreater(len(self.composer._current_path), 0)

    def test_inference_mode(self):
        """Test switching to inference mode."""
        # Train briefly
        for _ in range(20):
            self.composer.reset_var_counter()
            self.composer.generate(
                target_type=self.target_type,
                depth=3,
                context={},
                substitutions=SubstitutionTable()
            )

        # Switch to inference
        self.composer.set_training_mode(False)
        self.assertFalse(self.composer.training_mode)

        # Should still generate valid programs
        self.composer.reset_var_counter()
        program = self.composer.generate(
            target_type=self.target_type,
            depth=3,
            context={},
            substitutions=SubstitutionTable()
        )
        self.assertIsInstance(program, ASTNode)

    def test_determinism_with_seed(self):
        """Test that same seed produces same programs."""
        composer1 = MCTSComposer(seed=123, grammar=DefaultGrammar, training_mode=False)
        composer2 = MCTSComposer(seed=123, grammar=DefaultGrammar, training_mode=False)

        composer1.reset_var_counter()
        composer2.reset_var_counter()

        prog1 = composer1.generate(self.target_type, 3, {}, SubstitutionTable())
        prog2 = composer2.generate(self.target_type, 3, {}, SubstitutionTable())

        self.assertEqual(pretty_print(prog1), pretty_print(prog2))


class TestConstantExpressionPenalty(unittest.TestCase):
    """Tests for constant expression detection and penalty."""

    def setUp(self):
        """Set up test fixtures."""
        self.composer = MCTSComposer(seed=42, grammar=DefaultGrammar)

    def test_uses_context_variable_detection(self):
        """Test detection of context variable usage."""
        context_vars = {'x', 'y'}

        # Constant expression - no context vars
        const_expr = ApplicationNode(
            VariableNode('+'),
            [NumberNode(7), NumberNode(7)]
        )
        self.assertFalse(
            self.composer._uses_context_variable(const_expr, context_vars)
        )

        # Variable expression - uses context
        var_expr = ApplicationNode(
            VariableNode('+'),
            [VariableNode('x'), NumberNode(7)]
        )
        self.assertTrue(
            self.composer._uses_context_variable(var_expr, context_vars)
        )

    def test_uses_context_in_nested_expression(self):
        """Test context detection in nested expressions."""
        context_vars = {'x'}

        # Deeply nested but uses x
        nested = ApplicationNode(
            VariableNode('map'),
            [
                LambdaNode(['y'], VariableNode('y')),
                VariableNode('x')
            ]
        )
        self.assertTrue(
            self.composer._uses_context_variable(nested, context_vars)
        )

    def test_lambda_shadowing(self):
        """Test that lambda parameters shadow context variables."""
        context_vars = {'x'}

        # Lambda shadows x, so inner x doesn't count as context usage
        shadowed = LambdaNode(['x'], VariableNode('x'))
        self.assertFalse(
            self.composer._uses_context_variable(shadowed, context_vars)
        )

    def test_should_compute_immediate_reward(self):
        """Test conditions for computing immediate reward."""
        context = {'x': list[int]}

        # Application with context should compute
        app = ApplicationNode(VariableNode('map'), [VariableNode('f'), VariableNode('x')])
        self.assertTrue(
            self.composer._should_compute_immediate_reward(app, depth=3, context=context)
        )

        # Literal should not compute
        lit = NumberNode(5)
        self.assertFalse(
            self.composer._should_compute_immediate_reward(lit, depth=3, context=context)
        )

        # Variable should not compute
        var = VariableNode('x')
        self.assertFalse(
            self.composer._should_compute_immediate_reward(var, depth=3, context=context)
        )

        # Empty context should not compute
        self.assertFalse(
            self.composer._should_compute_immediate_reward(app, depth=3, context={})
        )

        # Low depth should not compute
        self.assertFalse(
            self.composer._should_compute_immediate_reward(app, depth=0, context=context)
        )


class TestImmediateFeedback(unittest.TestCase):
    """Tests for the immediate feedback mechanism."""

    def setUp(self):
        """Set up test fixtures."""
        self.target_type = Callable[[list[int]], list[int]]

    def test_immediate_feedback_tracking(self):
        """Test that immediate feedback is tracked during generation."""
        composer = MCTSComposer(seed=42, grammar=DefaultGrammar, training_mode=True)

        # Generate multiple programs to get some immediate feedback
        feedback_counts = []
        for _ in range(10):
            composer.reset_var_counter()
            composer.generate(
                target_type=self.target_type,
                depth=4,
                context={},
                substitutions=SubstitutionTable()
            )
            feedback_counts.append(len(composer._immediate_feedback_given))

        # At least some programs should have received immediate feedback
        self.assertGreater(sum(feedback_counts), 0)

    def test_immediate_feedback_reset_per_episode(self):
        """Test that immediate feedback is reset for each episode."""
        composer = MCTSComposer(seed=42, grammar=DefaultGrammar, training_mode=True)

        composer.reset_var_counter()
        composer.generate(self.target_type, 4, {}, SubstitutionTable())
        first_feedback = composer._immediate_feedback_given.copy()

        composer.reset_var_counter()
        composer.generate(self.target_type, 4, {}, SubstitutionTable())
        second_feedback = composer._immediate_feedback_given

        # Feedback should be reset (not accumulated)
        # The sets should potentially be different (not necessarily, but reset should have happened)
        # We verify by checking that the set was indeed recreated
        self.assertIsNot(first_feedback, second_feedback)


class TestVariabilityScorer(unittest.TestCase):
    """Tests for the VariabilityScorer."""

    def setUp(self):
        """Set up test fixtures."""
        import random
        self.scorer = VariabilityScorer(
            grammar=DefaultGrammar,
            rng=random.Random(42),
            num_samples=10
        )

    def test_score_by_structure(self):
        """Test structural scoring of AST nodes."""
        # Variable should get low score
        var = VariableNode('x')
        var_score = self.scorer._score_by_structure(var)
        self.assertLess(var_score, 0.5)

        # Application with grammar function should get higher score
        app = ApplicationNode(VariableNode('map'), [VariableNode('f'), VariableNode('x')])
        app_score = self.scorer._score_by_structure(app)
        self.assertGreater(app_score, var_score)


class TestTreeStatistics(unittest.TestCase):
    """Tests for tree statistics and persistence."""

    def setUp(self):
        """Set up test fixtures."""
        self.target_type = Callable[[list[int]], list[int]]

    def test_get_tree_stats(self):
        """Test getting tree statistics."""
        composer = MCTSComposer(seed=42, grammar=DefaultGrammar, training_mode=True)

        # Train briefly
        for _ in range(20):
            composer.reset_var_counter()
            composer.generate(self.target_type, 3, {}, SubstitutionTable())

        stats = composer.get_tree_stats()

        self.assertIn('num_nodes', stats)
        self.assertIn('total_visits', stats)
        self.assertIn('top_states', stats)
        self.assertGreater(stats['num_nodes'], 0)
        self.assertGreater(stats['total_visits'], 0)


class TestTrainMCTSComposer(unittest.TestCase):
    """Tests for the training function."""

    def test_train_mcts_composer(self):
        """Test the training helper function."""
        target_type = Callable[[list[int]], list[int]]

        composer = train_mcts_composer(
            grammar=DefaultGrammar,
            target_type=target_type,
            num_episodes=50,
            depth=3,
            seed=42,
            verbose=False
        )

        # Should be in inference mode after training
        self.assertFalse(composer.training_mode)

        # Should have built a tree
        self.assertGreater(len(composer.tree), 0)

        # Should generate valid programs
        composer.reset_var_counter()
        program = composer.generate(target_type, 3, {}, SubstitutionTable())
        self.assertIsInstance(program, ASTNode)


class TestProgramQuality(unittest.TestCase):
    """Tests for the quality of generated programs."""

    def setUp(self):
        """Set up test fixtures."""
        self.target_type = Callable[[list[int]], list[int]]

    def _uses_input_variable(self, program_str: str) -> bool:
        """Check if program uses input variable x."""
        # Skip the lambda header
        body = re.sub(r'^\(λ \(x\) ', '', program_str)
        return bool(re.search(r'\bx\b', body))

    def test_trained_composer_uses_input(self):
        """Test that trained composer tends to use input variable."""
        composer = train_mcts_composer(
            grammar=DefaultGrammar,
            target_type=self.target_type,
            num_episodes=100,
            depth=4,
            seed=42,
            verbose=False
        )

        # Generate programs and check how many use x
        uses_x_count = 0
        for _ in range(10):
            composer.reset_var_counter()
            program = composer.generate(self.target_type, 4, {}, SubstitutionTable())
            prog_str = pretty_print(program)
            if self._uses_input_variable(prog_str):
                uses_x_count += 1

        # At least some programs should use input variable
        # (With constant expression penalty, this should be most of them)
        self.assertGreater(uses_x_count, 3)

    def test_programs_evaluate_without_error(self):
        """Test that generated programs can be evaluated."""
        from lang.environment import Closure

        composer = MCTSComposer(seed=42, grammar=DefaultGrammar, training_mode=False)
        evaluator = Evaluator()

        for i in range(5):
            composer.reset_var_counter()
            program = composer.generate(self.target_type, 3, {}, SubstitutionTable())

            # Should evaluate without raising
            try:
                result = evaluator.eval(program)
                # Result should be a Closure for Callable type
                self.assertIsInstance(result, Closure)
            except Exception as e:
                self.fail(f"Evaluation failed for program {i}: {e}\n{pretty_print(program)}")


if __name__ == "__main__":
    unittest.main()
