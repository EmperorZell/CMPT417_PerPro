import time as timer

import numpy as np
from numpy.core.numeric import infty, Infinity

from single_agent_planner import compute_heuristics, a_star, get_sum_of_cost


class PrioritizedPlanningSolver(object):
    """A planner that plans for each robot sequentially."""

    def __init__(self, my_map, starts, goals):
        """my_map   - list of lists specifying obstacle positions
        starts      - [(x1, y1), (x2, y2), ...] list of start locations
        goals       - [(x1, y1), (x2, y2), ...] list of goal locations
        """

        self.my_map = my_map
        self.starts = starts
        self.goals = goals
        self.num_of_agents = len(goals)

        self.CPU_time = 0

        # compute heuristics for the low-level search
        self.heuristics = []
        for goal in self.goals:
            self.heuristics.append(compute_heuristics(my_map, goal))

    def find_solution(self):
        """ Finds paths for all agents from their start locations to their goal locations."""

        start_time = timer.time()
        result = []
        constraints = []

        for i in range(self.num_of_agents):  # Find path for each agent

            # Create a "worst" time for any agent by adding all the avaiable movement spaces, the times of the previous...
            # agents and the shortest distance to the goal from start

            # Saves the dimensions of the map
            rows, columns = len(self.my_map[1]), len(self.my_map[0])
            validSquares = sum(not self.my_map[i][j] for j in range(columns) for j in range(rows))
            distanceToGoal = self.heuristics[i][self.starts[i]]
            currentLineLength = sum(len(path) - 1 for path in result)

            skyIsTheLimit = validSquares + distanceToGoal + currentLineLength

            constraints.append({"agent": i, "maxTime": skyIsTheLimit})

            path = a_star(self.my_map, self.starts[i], self.goals[i], self.heuristics[i],
                          i, constraints)
            if path is None:
                raise BaseException('No solutions')
            result.append(path)

            ##############################
            # Task 2: Add constraints here
            #         Useful variables:
            #            * path contains the solution path of the current (i'th) agent, e.g., [(1,1),(1,2),(1,3)]
            #            * self.num_of_agents has the number of total agents
            #            * constraints: array of constraints to consider for future A* searches
            for j in range (i + 1, self.num_of_agents):
                for t, loc, in enumerate(path):
                    # Future agents can not be on the same vertex at the same time
                    constraints.append({"agent": j, "loc": [loc], "timestep": t})

                    # can not move with or through each other
                    currentLoc = loc

                    if t > 0:
                        prevLocation = path[t - 1]
                        constraints.append({"agent": j, "loc": [currentLoc, prevLocation], "timestep": t})

                # # might need to change chungus to something else like inf and add something to limit it if stuck forever
                # chungus = 100
                # goalLocation = path[len(path) - 1];
                # fishyCeiling = len(path) + chungus
                # for t in range(len(path), fishyCeiling):
                #     constraints.append({"agent": j, "loc": [goalLocation], "timestep": t})

            ##############################

        self.CPU_time = timer.time() - start_time

        print("\n Found a solution! \n")
        print("CPU time (s):    {:.2f}".format(self.CPU_time))
        print("Sum of costs:    {}".format(get_sum_of_cost(result)))
        print(result)
        return result
