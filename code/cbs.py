import time as timer
import heapq
import random

# from paths_violate_constraint import paths_violate_constraint
from single_agent_planner import compute_heuristics, a_star, get_location, get_sum_of_cost

def paths_violate_constraint(constraint, paths):
    assert constraint['positive'] is True
    rst = []
    for i in range(len(paths)):
        if i == constraint['agent']:
            continue
        curr = get_location(paths[i], constraint['timestep'])
        prev = get_location(paths[i], constraint['timestep'] - 1)
        if len(constraint['loc']) == 1:  # vertex constraint
            if constraint['loc'][0] == curr:
                rst.append(i)
        else:  # edge constraint
            if constraint['loc'][0] == prev or constraint['loc'][1] == curr \
                    or constraint['loc'] == [curr, prev]:
                rst.append(i)
    return rst


def detect_collision(path1, path2):
    ##############################
    # Task 3.1: Return the first collision that occurs between two robot paths (or None if there is no collision)
    #           There are two types of collisions: vertex collision and edge collision.
    #           A vertex collision occurs if both robots occupy the same location at the same timestep
    #           An edge collision occurs if the robots swap their location at the same timestep.
    #           You should use "get_location(path, t)" to get the location of a robot at time t.

    # first vertex collision
    T = max(len(path1), len(path2))
    for t in range(T):
        a1 = get_location(path1, t)
        a2 = get_location(path2, t)
        if a1 == a2:
            return {'loc': [a1], 'timestep': t}

    # first edge collision
    for t in range(1, T):
        prevLocA1, currLocA1 = get_location(path1, t-1), get_location(path1, t)
        prevLocA2, currLocA2 = get_location(path2, t-1), get_location(path2, t)
        if prevLocA1 == currLocA2 and prevLocA2 == currLocA1:
            return {'loc': [prevLocA1, currLocA1], 'timestep': t}

    return None


def detect_collisions(paths):
    ##############################
    # Task 3.1: Return a list of first collisions between all robot pairs.
    #           A collision can be represented as dictionary that contains the id of the two robots, the vertex or edge
    #           causing the collision, and the timestep at which the collision occurred.
    #           You should use your detect_collision function to find a collision between two robots.

    collisions = []
    n = len(paths)
    for i in range(n):
        for j in range(i+1, n):
            detectCollision = detect_collision(paths[i], paths[j])
            if detectCollision is not None:
                collisions.append({'a1': i, 'a2': j,
                                   'loc': detectCollision['loc'],
                                   'timestep': detectCollision['timestep']})
    return collisions


def standard_splitting(collision):
    ##############################
    # Task 3.2: Return a list of (two) constraints to resolve the given collision
    #           Vertex collision: the first constraint prevents the first agent to be at the specified location at the
    #                            specified timestep, and the second constraint prevents the second agent to be at the
    #                            specified location at the specified timestep.
    #           Edge collision: the first constraint prevents the first agent to traverse the specified edge at the
    #                          specified timestep, and the second constraint prevents the second agent to traverse the
    #                          specified edge at the specified timestep

    a1, a2 = collision['a1'], collision['a2']
    timestep = collision['timestep']
    loc = collision['loc']
    if len(loc) == 1: # vertex constraint
        v = loc[0]
        return [
            {'agent': a1, 'loc': [v], 'timestep': timestep},
            {'agent': a2, 'loc': [v], 'timestep': timestep},
        ]

    else:
        prev, curr = loc
        return [
            {'agent': a1, 'loc': [prev, curr], 'timestep': timestep},
            {'agent': a2, 'loc': [curr, prev], 'timestep': timestep},
        ]

def disjoint_splitting(collision):
    ##############################
    # Task 4.1: Return a list of (two) constraints to resolve the given collision
    #           Vertex collision: the first constraint enforces one agent to be at the specified location at the
    #                            specified timestep, and the second constraint prevents the same agent to be at the
    #                            same location at the timestep.
    #           Edge collision: the first constraint enforces one agent to traverse the specified edge at the
    #                          specified timestep, and the second constraint prevents the same agent to traverse the
    #                          specified edge at the specified timestep
    #           Choose the agent randomly

    theCoinOfAllTime = random.choice(['a1', 'a2'])
    aChoice = collision[theCoinOfAllTime]
    timestep = collision['timestep']
    loc = collision['loc']

    if len(loc) == 1: # vertex constraint
        v = loc[0]
        return [
            {'agent': aChoice, 'loc': [v], 'timestep': timestep},
            {'agent': aChoice, 'loc': [v], 'timestep': timestep, 'positive': True},
        ]

    else:
        prev, curr = loc
        if theCoinOfAllTime == 'a1':
            return [
                {'agent': aChoice, 'loc': [prev, curr], 'timestep': timestep},
                {'agent': aChoice, 'loc': [prev, curr], 'timestep': timestep, 'positive': True},
            ]
        return [
            {'agent': aChoice, 'loc': [curr, prev], 'timestep': timestep},
            {'agent': aChoice, 'loc': [curr, prev], 'timestep': timestep, 'positive': True},
        ]

class CBSSolver(object):
    """The high-level search of CBS."""

    def __init__(self, my_map, starts, goals):
        """my_map   - list of lists specifying obstacle positions
        starts      - [(x1, y1), (x2, y2), ...] list of start locations
        goals       - [(x1, y1), (x2, y2), ...] list of goal locations
        """

        self.my_map = my_map
        self.starts = starts
        self.goals = goals
        self.num_of_agents = len(goals)

        self.num_of_generated = 0
        self.num_of_expanded = 0
        self.CPU_time = 0

        self.open_list = []

        # compute heuristics for the low-level search
        self.heuristics = []
        for goal in self.goals:
            self.heuristics.append(compute_heuristics(my_map, goal))

    def push_node(self, node):
        heapq.heappush(self.open_list, (node['cost'], len(node['collisions']), self.num_of_generated, node))
        print("Generate node {}".format(self.num_of_generated))
        self.num_of_generated += 1

    def pop_node(self):
        _, _, id, node = heapq.heappop(self.open_list)
        print("Expand node {}".format(id))
        self.num_of_expanded += 1
        return node

    def find_solution(self, disjoint=True):
        """ Finds paths for all agents from their start locations to their goal locations

        disjoint    - use disjoint splitting or not
        """

        self.start_time = timer.time()

        # Generate the root node
        # constraints   - list of constraints
        # paths         - list of paths, one for each agent
        #               [[(x11, y11), (x12, y12), ...], [(x21, y21), (x22, y22), ...], ...]
        # collisions     - list of collisions in paths
        root = {'cost': 0,
                'constraints': [],
                'paths': [],
                'collisions': []}
        for i in range(self.num_of_agents):  # Find initial path for each agent
            path = a_star(self.my_map, self.starts[i], self.goals[i], self.heuristics[i],
                          i, root['constraints'])
            if path is None:
                raise BaseException('No solutions')
            root['paths'].append(path)

        root['cost'] = get_sum_of_cost(root['paths'])
        root['collisions'] = detect_collisions(root['paths'])
        self.push_node(root)

        # Task 3.1: Testing
        print(root['collisions'])

        # Task 3.2: Testing
        for collision in root['collisions']:
            print(standard_splitting(collision))

        ##############################
        # Task 3.3: High-Level Search
        #           Repeat the following as long as the open list is not empty:
        #             1. Get the next node from the open list (you can use self.pop_node())
        #             2. If this node has no collision, return solution
        #             3. Otherwise, choose the first collision and convert to a list of constraints (using your
        #                standard_splitting function). Add a new child node to your open list for each constraint
        #           Ensure to create a copy of any objects that your child nodes might inherit

        print("Using disjoint splitting:", disjoint)
        # disjoint = True
        while self.open_list:
            node = self.pop_node()

            if len(node['collisions']) == 0:
                self.print_results(node)
                return node['paths']

            collision = node['collisions'][0]
            newConstraints = (disjoint_splitting(collision) if disjoint
                else standard_splitting(collision))


            if not disjoint:
                for constraint in newConstraints:
                    child = {
                        'constraints': node['constraints'] + [constraint],
                        'paths': list(node['paths']),
                    }

                    i = constraint['agent']
                    replan = a_star(self.my_map, self.starts[i], self.goals[i], self.heuristics[i], i, child['constraints'])
                    if replan is None:
                        continue

                    child['paths'][i] = replan
                    child['cost'] = get_sum_of_cost(child['paths'])
                    child['collisions'] = detect_collisions(child['paths'])
                    self.push_node(child)

            else:
                for constraint in newConstraints:
                    child = {
                        'constraints': node['constraints'] + [constraint],
                        'paths': list(node['paths']),
                    }

                    if 'positive' in constraint and constraint['positive'] == True:
                        violatingAgents = paths_violate_constraint(constraint, node['paths'])
                        t = constraint['timestep']

                        if len(constraint['loc']) == 1:
                            for each in violatingAgents:
                                child['constraints'].append(
                                    {'agent': each,
                                     'loc': constraint['loc'],
                                     'timestep': t
                                     }
                                )

                        else:
                            for each in violatingAgents:
                                prev, curr = get_location(node['paths'][each], constraint['timestep']-1), get_location(node['paths'][each], constraint['timestep'])
                                child['constraints'].append(
                                    {'agent': each,
                                     'loc': [prev, curr],
                                     'timestep': t
                                     }
                                )

                        agentsToReplan = {constraint['agent']} | set(violatingAgents)

                    else:
                        agentsToReplan = {constraint['agent']}

                    deathToChildren = False
                    for agents in agentsToReplan:
                        replan = a_star(self.my_map,
                                        self.starts[agents],
                                        self.goals[agents],
                                        self.heuristics[agents],
                                        agents,
                                        child['constraints']
                                        )

                        if replan is None:
                            deathToChildren = True
                            break
                        child['paths'][agents] = replan

                    if deathToChildren:
                        continue

                    child['cost'] = get_sum_of_cost(child['paths'])
                    child['collisions'] = detect_collisions(child['paths'])
                    self.push_node(child)


        self.print_results(root)
        return root['paths']


    def print_results(self, node):
        print("\n Found a solution! \n")
        CPU_time = timer.time() - self.start_time
        print("CPU time (s):    {:.2f}".format(CPU_time))
        print("Sum of costs:    {}".format(get_sum_of_cost(node['paths'])))
        print("Expanded nodes:  {}".format(self.num_of_expanded))
        print("Generated nodes: {}".format(self.num_of_generated))
