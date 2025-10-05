import heapq

def move(loc, dir):
    directions = [(0, -1), (1, 0), (0, 1), (-1, 0)]
    return loc[0] + directions[dir][0], loc[1] + directions[dir][1]


def get_sum_of_cost(paths):
    rst = 0
    for path in paths:
        rst += len(path) - 1
    return rst


def compute_heuristics(my_map, goal):
    # Use Dijkstra to build a shortest-path tree rooted at the goal location
    open_list = []
    closed_list = dict()
    root = {'loc': goal, 'cost': 0}
    heapq.heappush(open_list, (root['cost'], goal, root))
    closed_list[goal] = root
    while len(open_list) > 0:
        (cost, loc, curr) = heapq.heappop(open_list)
        for dir in range(4):
            child_loc = move(loc, dir)
            child_cost = cost + 1
            if child_loc[0] < 0 or child_loc[0] >= len(my_map) \
               or child_loc[1] < 0 or child_loc[1] >= len(my_map[0]):
               continue
            if my_map[child_loc[0]][child_loc[1]]:
                continue
            child = {'loc': child_loc, 'cost': child_cost}
            if child_loc in closed_list:
                existing_node = closed_list[child_loc]
                if existing_node['cost'] > child_cost:
                    closed_list[child_loc] = child
                    # open_list.delete((existing_node['cost'], existing_node['loc'], existing_node))
                    heapq.heappush(open_list, (child_cost, child_loc, child))
            else:
                closed_list[child_loc] = child
                heapq.heappush(open_list, (child_cost, child_loc, child))

    # build the heuristics table
    h_values = dict()
    for loc, node in closed_list.items():
        h_values[loc] = node['cost']
    return h_values


def build_constraint_table(constraints, agent):
    ##############################
    # Task 1.2/1.3: Return a table that constains the list of constraints of
    #               the given agent for each time step. The table can be used
    #               for a more efficient constraint violation check in the 
    #               is_constrained function.

    table = {}
    for c in constraints or []:

        # If the current agent is the wrong agent for the constraints dont do anything
        if c["agent"] != agent:
            continue

        # Otherwise, save the timestep and the location
        t = c["timestep"]
        locs = c["loc"]

        entry = table.setdefault(t, {"vertex": set(), "edge": set()})
        if len(locs) == 1:
            entry["vertex"].add(tuple(locs[0])) # cant be here
        else:
            entry["edge"].add((tuple(locs[0]), tuple(locs[1]))) # cant do this
    return table


def get_location(path, time):
    if time < 0:
        return path[0]
    elif time < len(path):
        return path[time]
    else:
        return path[-1]  # wait at the goal location


def get_path(goal_node):
    path = []
    curr = goal_node
    while curr is not None:
        path.append(curr['loc'])
        curr = curr['parent']
    path.reverse()
    return path


def is_constrained(curr_loc, next_loc, next_time, constraint_table):
    ##############################
    # Task 1.2/1.3: Check if a move from curr_loc to next_loc at time step next_time violates
    #               any given constraint. For efficiency the constraints are indexed in a constraint_table
    #               by time step, see build_constraint_table.

    entry = constraint_table.get(next_time)
    if not entry:
        return False

    if tuple(next_loc) in entry["vertex"]:
        return True

    if (tuple(curr_loc), tuple(next_loc)) in entry["edge"]:
        return True

    return False


def push_node(open_list, node):
    heapq.heappush(open_list, (node['g_val'] + node['h_val'], node['h_val'], node['loc'], node))


def pop_node(open_list):
    _, _, _, curr = heapq.heappop(open_list)
    return curr


def compare_nodes(n1, n2):
    """Return true is n1 is better than n2."""
    return n1['g_val'] + n1['h_val'] < n2['g_val'] + n2['h_val']


def a_star(my_map, start_loc, goal_loc, h_values, agent, constraints):
    """ my_map      - binary obstacle map
        start_loc   - start position
        goal_loc    - goal position
        agent       - the agent that is being re-planned
        constraints - constraints defining where robot should or cannot go at each timestep
    """

    ##############################
    # Task 1.1: Extend the A* search to search in the space-time domain
    #           rather than space domain, only.

    # Checks if the next move is within bounds
    def in_bounds(loc):
        r, c = loc
        return 0 <= r < len(my_map) and 0 <= c < len(my_map[0])

    # Checks for walls
    def passable(loc):
        r, c = loc
        return not my_map[r][c]

    def neighbours_plus_wait(loc):
        r, c = loc
        # wait in place + 4-neighbours
        dirPlusWait = [(r, c), (r, c-1), (r+1, c), (r, c+1), (r-1, c)]
        validMoves = []
        for n in dirPlusWait:
            if in_bounds(n) and passable(n):
                validMoves.append(n)
        return validMoves

    constraintTable = build_constraint_table(constraints, agent)
    # earliestGoalTime = max(constraintTable.keys()) if constraintTable else 0
    lastTimeBlocked = -1
    for t, entry in constraintTable.items():
        if tuple(goal_loc) in entry["vertex"]:
            lastTimeBlocked = max(lastTimeBlocked, t)

    open_list = []
    closed_list = dict()
    # earliest_goal_timestep = 0
    h_value = h_values[start_loc]
    root = {'loc': start_loc, 't':0, 'g_val': 0, 'h_val': h_value, 'parent': None}
    push_node(open_list, root)
    closed_list[(root['loc'], root['t'])] = root

    while len(open_list) > 0:
        curr = pop_node(open_list)
        #############################
        # Task 1.4: Adjust the goal test condition to handle goal constraints
        # if curr['loc'] == goal_loc:
        #     return get_path(curr)

        if curr['loc'] == goal_loc and curr['t'] >= lastTimeBlocked:
            entry = constraintTable.get(curr['t'])
            if not entry or (tuple(goal_loc) not in entry['vertex']):
                return get_path(curr)

        for child_loc in neighbours_plus_wait(curr['loc']):
            # Compute next time
            child_t = curr['t'] + 1

            if is_constrained(curr['loc'], child_loc, child_t, constraintTable):
                continue

            child = {'loc': child_loc,
                     't': child_t,
                    'g_val': curr['g_val'] + 1,
                    'h_val': h_values[child_loc],
                    'parent': curr}

            key = (child['loc'], child['t'])

            if key in closed_list:
                existing_node = closed_list[key]

                if compare_nodes(child, existing_node):
                    closed_list[key] = child
                    push_node(open_list, child)

            else:
                closed_list[key] = child
                push_node(open_list, child)

    return None  # Failed to find solutions