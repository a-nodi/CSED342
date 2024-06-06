import collections, util, copy

SEED = 4532

############################################################
# Problem 1

# Hint: Take a look at the CSP class and the CSP examples in util.py
def create_chain_csp(n):
    # same domain for each variable
    domain = [0, 1]
    # name variables as x_1, x_2, ..., x_n
    variables = ['x%d'%i for i in range(1, n+1)]
    csp = util.CSP()
    # Problem 1a
    # BEGIN_YOUR_ANSWER
    def xor(a, b):
        return a ^ b

    for variable in variables:
        csp.add_variable(variable, domain)
    for i in range(0, n - 1):
        csp.add_binary_factor(variables[i], variables[i + 1], xor)
    # END_YOUR_ANSWER
    return csp


############################################################
# Problem 2

def create_nqueens_csp(n = 8):
    """
    Return an N-Queen problem on the board of size |n| * |n|.
    You should call csp.add_variable() and csp.add_binary_factor().

    @param n: number of queens, or the size of one dimension of the board.

    @return csp: A CSP problem with correctly configured factor tables
        such that it can be solved by a weighted CSP solver.
    """
    csp = util.CSP()
    # Problem 2a
    # BEGIN_YOUR_ANSWER

    domain = [i for i in range(0, n)]

    for i in range(0, n):
        csp.add_variable(f"queen_{i}", domain)

    # Index of queen works as x-coord of chess board
    # Value of queen works as y-coord of chess board
    # Constraint to solve this in O(n)
    # 1. y-coord of different queens should be different
    # 2. difference of x-coord and difference of y-coord should be different with two different queens 
    # (If same, two queens are in diagonal position)
    
    for i in range(0, n):
        for j in range(i + 1, n):
            csp.add_binary_factor(f"queen_{i}", f"queen_{j}", lambda queen1, queen2: queen1 != queen2)
            csp.add_binary_factor(f"queen_{i}", f"queen_{j}", lambda queen1, queen2: abs(queen1 - queen2) != j - i)

    # END_YOUR_ANSWER
    return csp

# A backtracking algorithm that solves weighted CSP.
# Usage:
#   search = BacktrackingSearch()
#   search.solve(csp)
class BacktrackingSearch():

    def reset_results(self):
        """
        This function resets the statistics of the different aspects of the
        CSP solver. We will be using the values here for grading, so please
        do not make any modification to these variables.
        """
        # Keep track of the best assignment and weight found.
        self.optimalAssignment = {}
        self.optimalWeight = 0

        # Keep track of the number of optimal assignments and assignments. These
        # two values should be identical when the CSP is unweighted or only has binary
        # weights.
        self.numOptimalAssignments = 0
        self.numAssignments = 0

        # Keep track of the number of times backtrack() gets called.
        self.numOperations = 0

        # Keep track of the number of operations to get to the very first successful
        # assignment (doesn't have to be optimal).
        self.firstAssignmentNumOperations = 0

        # List of all solutions found.
        self.allAssignments = []

    def print_stats(self):
        """
        Prints a message summarizing the outcome of the solver.
        """
        if self.optimalAssignment:
            print("Found %d optimal assignments with weight %f in %d operations" % \
                (self.numOptimalAssignments, self.optimalWeight, self.numOperations))
            print("First assignment took %d operations" % self.firstAssignmentNumOperations)
        else:
            print("No solution was found.")

    def get_delta_weight(self, assignment, var, val):
        """
        Given a CSP, a partial assignment, and a proposed new value for a variable,
        return the change of weights after assigning the variable with the proposed
        value.

        @param assignment: A dictionary of current assignment. Unassigned variables
            do not have entries, while an assigned variable has the assigned value
            as value in dictionary. e.g. if the domain of the variable A is [5,6],
            and 6 was assigned to it, then assignment[A] == 6.
        @param var: name of an unassigned variable.
        @param val: the proposed value.

        @return w: Change in weights as a result of the proposed assignment. This
            will be used as a multiplier on the current weight.
        """
        assert var not in assignment
        w = 1.0
        if self.csp.unaryFactors[var]:
            w *= self.csp.unaryFactors[var][val]
            if w == 0: return w
        for var2, factor in self.csp.binaryFactors[var].items():
            if var2 not in assignment: continue  # Not assigned yet
            w *= factor[val][assignment[var2]]
            if w == 0: return w
        return w

    def solve(self, csp, mcv = False, ac3 = False):
        """
        Solves the given weighted CSP using heuristics as specified in the
        parameter. Note that unlike a typical unweighted CSP where the search
        terminates when one solution is found, we want this function to find
        all possible assignments. The results are stored in the variables
        described in reset_result().

        @param csp: A weighted CSP.
        @param mcv: When enabled, Most Constrained Variable heuristics is used.
        @param ac3: When enabled, AC-3 will be used after each assignment of an
            variable is made.
        """
        # CSP to be solved.
        self.csp = csp

        # Set the search heuristics requested asked.
        self.mcv = mcv
        self.ac3 = ac3

        # Reset solutions from previous search.
        self.reset_results()

        # The dictionary of domains of every variable in the CSP.
        self.domains = {var: list(self.csp.values[var]) for var in self.csp.variables}

        # Perform backtracking search.
        self.backtrack({}, 0, 1)
        # Print summary of solutions.
        self.print_stats()

    def backtrack(self, assignment, numAssigned, weight):
        """
        Perform the back-tracking algorithms to find all possible solutions to
        the CSP.

        @param assignment: A dictionary of current assignment. Unassigned variables
            do not have entries, while an assigned variable has the assigned value
            as value in dictionary. e.g. if the domain of the variable A is [5,6],
            and 6 was assigned to it, then assignment[A] == 6.
        @param numAssigned: Number of currently assigned variables
        @param weight: The weight of the current partial assignment.
        """

        self.numOperations += 1
        assert weight > 0
        if numAssigned == self.csp.numVars:
            # A satisfiable solution have been found. Update the statistics.
            self.numAssignments += 1
            newAssignment = {}
            for var in self.csp.variables:
                newAssignment[var] = assignment[var]
            self.allAssignments.append(newAssignment)

            if len(self.optimalAssignment) == 0 or weight >= self.optimalWeight:
                if weight == self.optimalWeight:
                    self.numOptimalAssignments += 1
                else:
                    self.numOptimalAssignments = 1
                self.optimalWeight = weight

                self.optimalAssignment = newAssignment
                if self.firstAssignmentNumOperations == 0:
                    self.firstAssignmentNumOperations = self.numOperations
            return

        # Select the next variable to be assigned.
        var = self.get_unassigned_variable(assignment)
        # Get an ordering of the values.
        ordered_values = self.domains[var]

        # Continue the backtracking recursion using |var| and |ordered_values|.
        for val in ordered_values:
            deltaWeight = self.get_delta_weight(assignment, var, val)
            if deltaWeight > 0:
                assignment[var] = val
                self.backtrack(assignment, numAssigned + 1, weight * deltaWeight)
                del assignment[var]

    def get_unassigned_variable(self, assignment):
        """
        Given a partial assignment, return a currently unassigned variable.

        @param assignment: A dictionary of current assignment. This is the same as
            what you've seen so far.

        @return var: a currently unassigned variable.
        """

        if not self.mcv:
            # Select a variable without any heuristics.
            for var in self.csp.variables:
                if var not in assignment: return var
        else:
            # Problem 2b
            # Heuristic: most constrained variable (MCV)
            # Select a variable with the least number of remaining domain values.
            # Hint: given var, self.domains[var] gives you all the possible values
            # Hint: get_delta_weight gives the change in weights given a partial
            #       assignment, a variable, and a proposed value to this variable
            # Hint: for ties, choose the variable with lowest index in self.csp.variables
            # BEGIN_YOUR_ANSWER
            def is_not_assigned(_variable):
                return _variable not in assignment

            # Filter unassigned variable
            list_of_unassigned_variable = filter(is_not_assigned, [variable for variable in self.csp.variables])
            list_of_variable_score_pair = []

            # Index and variable name
            for variable in list_of_unassigned_variable:
                score = sum([self.get_delta_weight(assignment, variable, value) for value in self.domains[variable]])  # Calculate score of current variable
                list_of_variable_score_pair.append((variable, score))  # Store score with coresponding variable

            # Argmin variable
            min_score_variable = min(list_of_variable_score_pair, key=lambda pair: pair[1])[0]

            return min_score_variable            
            # END_YOUR_ANSWER


############################################################
# Problem 3

def get_sum_variable(csp, name, variables, maxSum):
    """
    Given a list of |variables| each with non-negative integer domains,
    returns the name of a new variable with domain range(0, maxSum+1), such that
    it's consistent with the value |n| iff the assignments for |variables|
    sums to |n|.

    @param name: Prefix of all the variables that are going to be added.
        Can be any hashable objects. For every variable |var| added in this
        function, it's recommended to use a naming strategy such as
        ('sum', |name|, |var|) to avoid conflicts with other variable names.
    @param variables: A list of variables that are already in the CSP that
        have non-negative integer values as its domain.
    @param maxSum: An integer indicating the maximum sum value allowed. You
        can use it to get the auxiliary variables' domain

    @return result: The name of a newly created variable with domain range
        [0, maxSum] such that it's consistent with an assignment of |n|
        iff the assignment of |variables| sums to |n|.
    """
    # Problem 3a
    # BEGIN_YOUR_ANSWER
    PREVIOUS, CURRENT = 0, 1

    def is_aux_pair_valid(_previous_aux_variable, _current_aux_variable):
        return _previous_aux_variable[CURRENT] == _current_aux_variable[PREVIOUS]
    
    def is_summation_valid(_variable, _aux_variable):
        return _aux_variable[CURRENT] - _aux_variable[PREVIOUS] == _variable

    def is_result(_result, _last_aux_variable):
        return _result == _last_aux_variable[CURRENT]

    # Result aux variable
    result = ('sum', name, 'result')

    # Edge case, Enforce the result value be 0 
    if len(variables) == 0:
        csp.add_variable(result, [0])
        return result
    
    # Add result aux varable
    csp.add_variable(result, list(range(0, maxSum + 1)))

    # Pair-wise domain for adding up variables sequently
    pair_domain = [(previous_aux_value, current_aux_value) for previous_aux_value in range(maxSum + 1) for current_aux_value in range(maxSum + 1)]

    # Add aux variables & constraints
    for i, variable in enumerate(variables):
        # Create and add variable
        aux_variable = ('sum', name, i)
        
        # Compress domain of first variable to be efficient
        if i == 0:
            csp.add_variable(aux_variable, [(0, value) for value in csp.values[variable[0]]])
        else:
            csp.add_variable(aux_variable, pair_domain)
        
        # Add constraint
        csp.add_binary_factor(variable, aux_variable, is_summation_valid)  # Summation constraint. Aux variables will have values related to variable's value
    
    # Pair up neighbourhood aux variable's values.
    for previous, current in zip(list(range(0, len(variables) - 1)), list(range(1, len(variables)))):
        csp.add_binary_factor(('sum', name, previous), ('sum', name, current), is_aux_pair_valid)
    
    # Set result value
    csp.add_binary_factor(result, ("sum", name, len(variables) - 1), is_result)
    
    return result
    # END_YOUR_ANSWER

def create_lightbulb_csp(buttonSets, numButtons):
    """
    Return an light-bulb problem for the given buttonSets.
    You can exploit get_sum_variable().

    @param buttonSets: buttonSets is a tuple of sets of buttons. buttonSets[i] is a set including all indices of buttons which toggle the i-th light bulb.
    @param numButtons: the number of all buttons

    @return csp: A CSP problem with correctly configured factor tables
        such that it can be solved by a weighted CSP solver.
    """
    numBulbs = len(buttonSets)
    csp = util.CSP()

    assert all(all(0 <= buttonIndex < numButtons
                   for buttonIndex in buttonSet)
               for buttonSet in buttonSets)

    # Problem 3b
    # BEGIN_YOUR_ANSWER
    ON, OFF = True, False
    PREVIOUS, CURRENT = 0, 1
    
    def is_pair_valid(_previous_aux_variable, _current_aux_variable):
        return _previous_aux_variable[CURRENT] == _current_aux_variable[PREVIOUS]
    
    def is_toggling_valid(state, pair): 
        return pair[CURRENT] == (pair[PREVIOUS] == state)

    def is_result(_result, _last_aux_variable):
        return _result == _last_aux_variable[CURRENT]

    # Domain
    domain = [OFF, ON]
    # Pair-wise domain for toggling variables sequently
    pair_domain = [(OFF, OFF), (OFF, ON), (ON, OFF), (ON, ON)]

    for button_index in range(numButtons): 
        csp.add_variable(button_index, [OFF, ON])  # Add button as variable
    
    for bulb_index in range(numBulbs):
        # Result aux variable
        result = ('toggle', bulb_index, "currently_toggle_applied")
        csp.add_variable(result, domain)
        
        if len(buttonSets[bulb_index]) == 0:  # Edge case: No button is connected to current bulb
            csp.add_unary_factor(result, lambda state: state == OFF)

        else:
            # Create and add variable
            for i, buttonIndex in enumerate(buttonSets[bulb_index]):
                aux_variable = ('toggle', bulb_index, i)
                csp.add_variable(aux_variable, pair_domain)
                csp.add_binary_factor(buttonIndex, aux_variable, is_toggling_valid)  # Toggling constraint. Aux variables will have values related to variable's value
            
            # Prev bulb of first bulb doesn't exist
            csp.add_unary_factor(('toggle', bulb_index, 0), lambda aux: aux[PREVIOUS] == OFF)            
            
            # Pair up neighbourhood aux variable's values.
            for previous, current in zip(list(range(0, len(buttonSets[bulb_index]) - 1)), list(range(1, len(buttonSets[bulb_index])))):
                csp.add_binary_factor(('toggle', bulb_index, previous), ('toggle', bulb_index, current), is_pair_valid)

            # Set result value
            csp.add_binary_factor(result, ('toggle', bulb_index, len(buttonSets[bulb_index]) - 1), is_result)
        
        # Current bulb should be turned on
        csp.add_unary_factor(result, lambda state: state == ON)
    # END_YOUR_ANSWER
    return csp
