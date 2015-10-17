from rlglue.agent.Agent import Agent
from rlglue.agent import AgentLoader as AgentLoader
from rlglue.types import Action
from rlglue.utils import TaskSpecVRLGLUE3

import logging

import numpy as np
from sklearn import linear_model

from spectral_dagger.function_approximation import RectangularTileCoding
from spectral_dagger.function_approximation import StateActionFeatureExtractor
from spectral_dagger.utils.math import p_sequence
from spectral_dagger.mdp import LinearGibbsPolicy



class eNAC(Agent):

    def agent_init(self, task_spec_string):
        task_spec = TaskSpecVRLGLUE3.TaskSpecParser(task_spec_string)
        if task_spec.valid:
            assert len(task_spec.getIntObservations()) == 0, "expecting 1-dimensional discrete observations"
            assert len(task_spec.getDoubleObservations()) == 2, "expecting no continuous observations"

            for o in task_spec.getDoubleObservations():
                assert not task_spec.isSpecial(o[0]), " expecting min observation to be a number not a special value"
                assert not task_spec.isSpecial(o[1]), " expecting max observation to be a number not a special value"

            assert len(task_spec.getIntActions()) == 1, "expecting 1-dimensional discrete actions"
            assert len(task_spec.getDoubleActions()) == 0, "expecting no continuous actions"
            assert not task_spec.isSpecial(task_spec.getIntActions()[0][0]), " expecting min action to be a number not a special value"
            assert not task_spec.isSpecial(task_spec.getIntActions()[0][1]), " expecting max action to be a number not a special value"
            self.n_actions = task_spec.getIntActions()[0][1] + 1

        else:
            print "Task Spec could not be parsed: " + task_spec_string

        self.log_file_name = 'eNAC.log'
        self.logger = logging.getLogger("eNAC")
        self.logger.setLevel(logging.INFO)
        self.logger.addHandler(logging.FileHandler(
            filename=self.log_file_name, mode='w'))
        self.logger.info("Received task spec: %s", task_spec.ts)

        self.policy_frozen = False

        self.n_actions = task_spec.getIntActions()[0][1] + 1
        self.actions = range(self.n_actions)
        self.gamma = task_spec.getDiscountFactor()

        # Create feature extractor
        origin = [i[0] for i in task_spec.getDoubleObservations()]
        extent = np.array(
            [i[1] - i[0] for i in task_spec.getDoubleObservations()])

        state_feature_extractor = RectangularTileCoding(
            n_tilings=3, origin=origin, extent=extent,
            tile_counts=5*np.ones(2))
        self.feature_extractor = StateActionFeatureExtractor(
            state_feature_extractor, self.n_actions)
        self.n_features = self.feature_extractor.n_features

        theta = np.zeros(self.n_features)
        self.policy = LinearGibbsPolicy(
            self.actions, self.feature_extractor, theta)

        self.episode = 0
        self.time_step = 0

        self.trajectories_per_estimate = 3
        self.trajectory_batch = []
        self.current_trajectory = None
        self.gradient = np.zeros(self.n_features)
        self.mu = 0.2

        self.average_returns = []

        self._alpha = p_sequence(start=3., p=0.6)
        self.alpha = self._alpha.next()

    def agent_start(self, observation):

        if self.current_trajectory is not None:
            # If trajectory ends because of a time-out,
            # then agent_end is not called by rl_glue
            self.record_trajectory()

        self.episode += 1
        self.time_step = 0

        current_state = observation.doubleArray[0:2]
        self.policy.update(action=None, state=current_state)
        current_action = self.policy.get_action()

        action = Action()
        action.intArray = [current_action]

        self.logger.info("*" * 20 + "Episode: %s" % self.episode)
        self.logger.info("Agent received state: %s" % current_state)
        self.logger.info("Agent took action: %s" % current_action)
        self.logger.info("Agent alpha: %s" % self.alpha)

        return action

    def agent_step(self, reward, observation):
        self.time_step += 1

        current_state = observation.doubleArray[0:2]
        self.policy.update(action=None, state=current_state)
        current_action = self.policy.get_action()

        self.current_trajectory.features += (self.gamma ** self.time_step) * (
            self.policy.gradient_log(current_state, current_action))
        self.current_trajectory.total_reward += reward

        # Need to collect: summed discounted features.
        action = Action()
        action.intArray = [current_action]

        self.logger.info("Agent received reward: %s" % reward)
        self.logger.info("*" * 10 + "Time step: %s" % self.time_step)
        self.logger.info("Agent received state: %s" % current_state)
        self.logger.info("Agent took action: %s" % current_action)
        self.logger.info("Agent alpha: %s" % self.alpha)

        return action

    def agent_end(self, reward):
        self.current_trajectory.total_reward += reward

        self.record_trajectory()

    def record_trajectory(self):
        assert self.current_trajectory is not None, "Empty trajectory."

        self.trajectory_batch.append(self.current_trajectory)

        self.current_trajectory = None

        if len(self.trajectory_batch) == self.trajectories_per_estimate:

            norm = np.linalg.norm(gradient)
            if norm > 0:
                gradient /= norm

            # implement momentum
            self.gradient = self.mu * self.gradient + gradient

            self.policy.theta += self.alpha * self.gradient

            self.logger.info("*" * 20 + "Time step: %s" % self.time_step)
            self.logger.info("Gradient step")
            self.logger.info("Step norm: %s" % norm)
            self.logger.info("Alpha: %s" % self.alpha)

            if hasattr(self._alpha, 'next'):
                try:
                    next_alpha = self._alpha.next()
                    self.alpha = next_alpha
                except StopIteration:
                    pass
REINFORCE