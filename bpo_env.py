import gymnasium as gym
from gymnasium import spaces, Env
import random
import numpy as np
from typing import List

from simulator import Simulator
import warnings

warnings.filterwarnings("ignore")

class BPOEnv(Env):
    def __init__(self, running_time, config_type, allow_postponing=True, reward_function=None, postpone_penalty=0, write_to=None) -> None:
        self._print = False
        self.running_time = running_time
        self.counter = 0
        self.nr_bad_assignments = 0
        self.nr_postpone = 0
        self.config_type = config_type
        self.allow_postponing = allow_postponing
        self.action_mask_limit = 1 if self.allow_postponing else 0

        self.reward_function = reward_function
        self.postpone_penalty = postpone_penalty
        self.write_to = write_to
        self.action_number = [0, 0, 0]
        self.action_time = [0, 0, 0]
        self.step_print = False
        self.last_reward = 0
        self.additional_rewards = 0
        self.previous_reward_time = 0

        # Parameters for case by case assignment
        self.nr_other_cases = 5
        self.considered_cases = 0
        self.step_count = 0

        self.simulator = Simulator(running_time=self.running_time, planner=None, config_type=self.config_type, allow_postponing=self.allow_postponing, reward_function=self.reward_function, write_to=self.write_to)
        #print(self.simulator.input)
        #print(self.simulator.output)
        #define lows and highs for different sections of the input
        lows = np.array([0 for x in range(len(self.simulator.input))])
        highs = np.array([1 for x in range(len(self.simulator.input))])  
        
        self.observation_space = spaces.Box(low=lows,
                                            high=highs,
                                            shape=(len(self.simulator.input),), dtype=np.float64) #observation space is the cartesian product of resources and tasks
        
        ## spaces.Discrete returns a number between 0 and len(self.simulator.output)
        self.action_space = spaces.Discrete(len(self.simulator.output)) # Action space is all possible assignments + postpone action

    def step(self, action):
        self.step_count += 1
        if self._print:

            print(f'\n step: {self.step_count} @ time: {self.simulator.now}')
            task_types_num = [sum([1.0 if task.task_type == el else 0.0 for task in self.simulator.available_tasks]) for el in self.simulator.task_types if el != "Start"]
            print(f"Uncompleted cases: {len(self.simulator.uncompleted_cases)}")
            print(f"Available resources: {self.simulator.available_resources}. Available tasks: {task_types_num}")
            mask = self.simulator.define_action_masks(self.considered_cases)
            print(list(zip(self.simulator.output, mask)))
        # 1 Process action
        # 2 Do the timestep
        # 3 Return reward
        # Assign one resources per iteration. If possible, another is assigned in next step without advancing simulator
        assignment = self.simulator.output[action] # (Resource X, Task Y)
        next_case_mask = 1 if self.considered_cases < len(self.simulator.uncompleted_cases) - 1 else 0
        action_mask_limit = self.action_mask_limit + next_case_mask
        if self._print: print(f"Assignment: {assignment}")
        #print(assignment, self.simulator.state)
        if assignment != 'Postpone' and assignment != 'Next_case':
            self.considered_cases = 0
            assignment = (assignment[0], (next((x for x in self.simulator.available_tasks if x.task_type == assignment[1]), None)))
            self.simulator.process_assignment(assignment)
            if self._print: 
                print('after')
                task_types_num = [sum([1.0 if task.task_type == el else 0.0 for task in self.simulator.available_tasks]) for el in self.simulator.task_types if el != "Start"]
                print(f"Uncompleted cases: {len(self.simulator.uncompleted_cases)}")
                print(f"Available resources: {self.simulator.available_resources}. Available tasks: {task_types_num}")

            while (sum(self.simulator.define_action_masks(self.considered_cases)) <= action_mask_limit) and (self.simulator.status != 'FINISHED'):
                self.simulator.run(considered_cases=self.considered_cases)
        elif assignment == 'Next_case':
            self.considered_cases += 1
            if self.considered_cases >= len(self.simulator.uncompleted_cases):
                self.considered_cases = 0
            if self._print: print(f"Next case: {self.considered_cases}")
        elif assignment == 'Postpone': # Postpone
            self.considered_cases = 0
            self.simulator.current_reward -= self.postpone_penalty # In case you want to penalize the agent for postponing. Default = 0
            # Generate two arrays to check if the simulator state changes
            unassigned_tasks =  [sum([1 if task.task_type == el else 0 for task in self.simulator.available_tasks]) for el in self.simulator.task_types] 
            unassigned_tasks_compare = [task_sum for task_sum in unassigned_tasks]
            #print(unassigned_tasks)
            available_resources = [resource for resource in self.simulator.available_resources]
            available_resources_compare = [resource for resource in available_resources]
            # Keep running the simulator until the state changes or the termination condition is reached
            while (self.simulator.status != 'FINISHED') and ((sum(self.simulator.define_action_masks(self.considered_cases)) <= action_mask_limit) or (unassigned_tasks == unassigned_tasks_compare and \
                    available_resources == available_resources_compare)):
                self.simulator.run(self.considered_cases) # Run until next decision epoch

                unassigned_tasks_compare = [sum([1 if task.task_type == el else 0 for task in self.simulator.available_tasks]) for el in self.simulator.task_types]
                available_resources_compare = [resource for resource in self.simulator.available_resources]

        reward = self.simulator.current_reward # Update reward
        self.simulator.current_reward = 0

        if self.simulator.status == 'FINISHED':
            return self.simulator.get_state(self.considered_cases, self.nr_other_cases), reward, True, {}, {}
        else:
            return self.simulator.get_state(self.considered_cases, self.nr_other_cases), reward, False, {}, {}


    def reset(self, seed: int | None = None):
        """Resets the environment to an initial state and returns an initial observation.

        Note that this function should not reset the environment's random
        number generator(s); random variables in the environment's state should
        be sampled independently between multiple calls to `reset()`. In other
        words, each call of `reset()` should yield an environment suitable for
        a new episode, independent of previous episodes.

        Returns:
            observation (object): the initial observation.
        """

        #print("-------Resetting environment-------")
        #print(f"Step count before reset: {self.step_count}")
        #print(self.action_masks())
        unassigned_tasks = [sum([1 if task.task_type == el else 0 for task in self.simulator.available_tasks]) for el in self.simulator.task_types]
        #print(f"Unassigned Tasks (before): {unassigned_tasks}")
        # Reinitialize the environment
        self.__init__(self.running_time, self.config_type, allow_postponing=self.allow_postponing, 
                    reward_function=self.reward_function, postpone_penalty=self.postpone_penalty, 
                    write_to=self.write_to)
        
        next_case_mask = 1 if self.considered_cases < len(self.simulator.uncompleted_cases) - 1 else 0
        action_mask_limit = self.action_mask_limit + next_case_mask

        # Get initial tasks and resources state
        unassigned_tasks = [sum([1 if task.task_type == el else 0 for task in self.simulator.available_tasks]) for el in self.simulator.task_types]
        unassigned_tasks_compare = [task_sum for task_sum in unassigned_tasks]
        available_resources = [resource for resource in self.simulator.available_resources]
        available_resources_compare = [resource for resource in available_resources]

        # Keep running the simulator until the state changes or the termination condition is reached
        while (self.simulator.status != 'FINISHED') and ((sum(self.simulator.define_action_masks(self.considered_cases)) <= action_mask_limit) or 
                                                        (unassigned_tasks == unassigned_tasks_compare and available_resources == available_resources_compare)):
            #print(f"Simulator Status: {self.simulator.status}")
            #print(f"Action Masks: {self.simulator.define_action_masks(self.considered_cases)}")
            #print(f"Unassigned Tasks (before): {unassigned_tasks}")
            #print(f"Available Resources (before): {available_resources}")

            self.simulator.run(self.considered_cases)  # Run until next decision epoch

            # Update task and resource states for comparison
            unassigned_tasks_compare = [sum([1 if task.task_type == el else 0 for task in self.simulator.available_tasks]) for el in self.simulator.task_types]
            available_resources_compare = [resource for resource in self.simulator.available_resources]

            #print(f"Unassigned Tasks (after): {unassigned_tasks_compare}")
            #print(f"Available Resources (after): {available_resources_compare}")

        print("-------Environment reset-------")
        return self.simulator.get_state(self.considered_cases, self.nr_other_cases), {}




        # while (sum(self.simulator.define_action_masks(self.considered_cases)) <= action_mask_limit):
        #     print( 'here reset')
        #     print(next_case_mask)
        #     print(action_mask_limit, self.considered_cases, len(self.simulator.uncompleted_cases))
        #     self.simulator.run(self.considered_cases) # Run the simulator to get to the first decision epoch
        print("-------Environment reset-------")
        return self.simulator.get_state(self.considered_cases, self.nr_other_cases), {}


    def render(self, mode='human', close=False):
        print(f"Average reward: {self.average_cycle_time}")


    def action_masks(self) -> List[bool]:
        return self.simulator.define_action_masks(self.considered_cases)