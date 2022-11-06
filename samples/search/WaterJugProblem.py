
# coding=utf-8
from __future__ import print_function
from simpleai.search import SearchProblem
from simpleai.search.traditional import breadth_first, depth_first, limited_depth_first, iterative_limited_depth_first, uniform_cost
#from simpleai.search.viewers import WebViewer
#from simpleai.search.viewers import ConsoleViewer
from simpleai.search.viewers import BaseViewer
capacities = [0,0,0]
targets = [0,0,0]

class WaterJugProblem(SearchProblem):
    costOfAlgorithm = 0
    def __init__(self, initial_state=None, graph_search=None, viewer=None,  depth_limit=None):
        self.initial_state = initial_state
        self.graph_search = graph_search
        self.depth_limit =  depth_limit
        super(WaterJugProblem, self).__init__(initial_state=(0, 0, 0))
        is_grap_search = self.graph_search
        for i in range(0,3):
            capacities[i]=(int(input("Please enter capacity of Jug{0}: ".format(i+1))))
        for i in range(0, 3):
            targets[i]=(int(input("Please enter target of Jug{0}: ".format(i+1))))
        print("\n****************************")
        print("Capacity: ", capacities,"\nTarget: ", targets ,"\nInitial: ", initial_state, )
        print("****************************")
        print("\n\n *** Uninformed Search Algorithms ***")
        print("\n***************\nGraph Search? ",graph_search)

    def actions(self, state):
        return ["Fill Jug1", "Fill Jug2", "Fill Jug3", "Empty Jug1", "Empty Jug2", "Empty Jug3", "Pour Jug1 to Jug2", "Pour Jug1 to Jug3", "Pour Jug2 to Jug1", "Pour Jug2 to Jug3",
                "Pour Jug3 to Jug1", "Pour Jug3 to Jug2"]

    def result(self, state, action):
        next_state = list(state)

        def PourToOther(a, b):
            next_state[b] = min(state[a] + state[b], capacities[b])
            next_state[a] = state[a] - (next_state[b] - state[b])

        if action == "Fill Jug1":
            next_state[0] = capacities[0]
        elif action == "Fill Jug2":
            next_state[1] = capacities[1]
        elif action == "Fill Jug3":
            next_state[2] = capacities[2]
        elif action == "Empty Jug1":
            next_state[0] = 0
        elif action == "Empty Jug2":
            next_state[1] = 0
        elif action == "Empty Jug3":
            next_state[2] = 0
        elif action == "Pour Jug1 to Jug2":
            amountOfTrans = min(state[0], capacities[1] - state[1])
            next_state[0] = state[0] - amountOfTrans
            next_state[1] = state[1] + amountOfTrans
        elif action == "Pour Jug1 to Jug3":
            amountOfTrans = min(state[0], capacities[2] - state[2])
            next_state[0] = state[0] - amountOfTrans
            next_state[2] = state[2] + amountOfTrans
        elif action == "Pour Jug2 to Jug1":
            amountOfTrans = min(state[1], capacities[0] - state[0])
            next_state[1] = state[1] - amountOfTrans
            next_state[0] = state[0] + amountOfTrans
        elif action == "Pour Jug2 to Jug3":
            amountOfTrans = min(state[1], capacities[2] - state[2])
            next_state[1] = state[1] - amountOfTrans
            next_state[2] = state[2] + amountOfTrans
        elif action == "Pour Jug3 to Jug1":
            amountOfTrans = min(state[2], capacities[0] - state[0])
            next_state[2] = state[2] - amountOfTrans
            next_state[0] = state[0] + amountOfTrans
        elif action == "Pour Jug3 to Jug2":
            amountOfTrans = min(state[2], capacities[1] - state[1])
            next_state[2] = state[2] - amountOfTrans
            next_state[1] = state[1] + amountOfTrans

        return tuple(next_state)


    def cost(self, state, action, state2):
        self.costOfAlgorithm+=1
        return 1

    def is_goal(self, state):
       return bool(list(state) == targets)

    def value(self, state):
        '''Returns the value of `state` as it is needed by optimization
           problems.
           Value is a number (integer or floating point).'''
        raise NotImplementedError

    def heuristic(self, state):
        '''Returns an estimate of the cost remaining to reach the solution
           from `state`.'''
        return 0

    def crossover(self, state1, state2):
        """
        Crossover method for genetic search. It should return a new state that
        is the 'mix' (somehow) of `state1` and `state2`.
        """
        raise NotImplementedError

    def mutate(self, state):
        """
        Mutation method for genetic search. It should return a new state that
        is a slight random variation of `state`.
        """
        raise NotImplementedError

    def generate_random_state(self):
        """
        Generates a random state for genetic search. It's mainly used for the
        seed states in the initilization of genetic search.
        """
        raise NotImplementedError

    def state_representation(self, state):
        """
        Returns a string representation of a state.
        By default it returns str(state).
        """
        return str(state)

    def action_representation(self, action):
        """
        Returns a string representation of an action.
        By default it returns str(action).
        """
        return str(action)


class SearchNode(object):
    '''Node of a search process.'''

    def __init__(self, state, parent=None, action=None, cost=0, problem=None,
                 depth=0):
        self.state = state
        self.parent = parent
        self.action = action
        self.cost = cost
        self.problem = problem or parent.problem
        self.depth = depth

    def expand(self, local_search=False):
        '''Create successors.'''
        new_nodes = []
        for action in self.problem.actions(self.state):
            new_state = self.problem.result(self.state, action)
            cost = self.problem.cost(self.state,
                                     action,
                                     new_state)
            nodefactory = self.__class__
            new_nodes.append(nodefactory(state=new_state,
                                         parent=None if local_search else self,
                                         problem=self.problem,
                                         action=action,
                                         cost=self.cost + cost,
                                         depth=self.depth + 1))
        return new_nodes

    def path(self):
        '''Path (list of nodes and actions) from root to this node.'''
        node = self
        path = []
        while node:
            path.append((node.action, node.state))
            node = node.parent
        return list(reversed(path))

    def __eq__(self, other):
        return isinstance(other, SearchNode) and self.state == other.state

    def state_representation(self):
        return self.problem.state_representation(self.state)

    def action_representation(self):
        return self.problem.action_representation(self.action)

    def __repr__(self):
        return 'Node <%s>' % self.state_representation().replace('\n', ' ')

    def __hash__(self):
        return hash((
            self.state,
            self.parent,
            self.action,
            self.cost,
            self.depth,
        ))


class SearchNodeCostOrdered(SearchNode):
    def __lt__(self, other):
        return self.cost < other.cost


class SearchNodeValueOrdered(SearchNode):
    def __init__(self, *args, **kwargs):
        super(SearchNodeValueOrdered, self).__init__(*args, **kwargs)
        self.value = self.problem.value(self.state)

    def __lt__(self, other):
        # value must work inverted, because heapq sorts 1-9
        # and we need 9-1 sorting
        return -self.value < -other.value


class SearchNodeHeuristicOrdered(SearchNode):
    def __init__(self, *args, **kwargs):
        super(SearchNodeHeuristicOrdered, self).__init__(*args, **kwargs)
        self.heuristic = self.problem.heuristic(self.state)

    def __lt__(self, other):
        return self.heuristic < other.heuristic


class SearchNodeStarOrdered(SearchNodeHeuristicOrdered):
    def __lt__(self, other):
        return self.heuristic + self.cost < other.heuristic + other.cost


class CspProblem(object):
    def __init__(self, variables, domains, constraints):
        self.variables = variables
        self.domains = domains
        self.constraints = constraints

        # variable-based constraints dict
        self.var_contraints = dict([(v, [constraint
                                         for constraint in constraints
                                         if v in constraint[0]])
                                    for v in variables])

        # calculate degree of each variable
        self.var_degrees = dict([(v, len(self.var_contraints[v]))
                                 for v in variables])


def test():
    searchers = [breadth_first, depth_first, limited_depth_first, uniform_cost, iterative_limited_depth_first]
    #searchers = [breadth_first]
   # my_viewer = WebViewer()
    problems = [WaterJugProblem(initial_state=(0, 0, 0), graph_search=True, viewer = None )] #As the number of calls to the class increases
    #the number of problems will also increase, for now, we have called it once.
    for p in problems:
        for s in searchers:
            my_viewer = BaseViewer()
            print("Applying " , s.__name__)
            if s.__name__ == "limited_depth_first":
                result = s(p, graph_search = True, depth_limit=None, viewer= my_viewer)
            else: result = s(p, graph_search = True, viewer= my_viewer)
            print("Resulting state: ", result.state)
            print("Resulting path:")
            for i in range(len(result.path())):
                print(i," . ", result.path()[i])
            print("Total cost: ", len(result.path())-1)
            print("Viewer stats: \n", "max_fringe_size: ", my_viewer.stats, "\n\n" )


test()