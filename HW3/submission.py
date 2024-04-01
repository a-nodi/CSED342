from typing import Callable, List, Set

import shell
import util
import wordsegUtil


############################################################
# Problem 1a: Solve the segmentation problem under a unigram model

class WordSegmentationProblem(util.SearchProblem):
    def __init__(self, query: str, unigramCost: Callable[[str], float]):
        self.query = query
        self.unigramCost = unigramCost

    def startState(self):
        # BEGIN_YOUR_CODE (our solution is 1 line of code, but don't worry if you deviate from this)
        return self.query
        # END_YOUR_CODE

    def isEnd(self, state) -> bool:
        # BEGIN_YOUR_CODE (our solution is 1 lines of code, but don't worry if you deviate from this)
        return len(state) == 0
        # END_YOUR_CODE

    def succAndCost(self, state):
        # BEGIN_YOUR_CODE (our solution is 2 lines of code, but don't worry if you deviate from this)
        list_of_succ_and_cost = []
        
        # Construct Successor, cost pair
        for i in range(len(state), 0, -1):
            action = state[:i]  # Current state (target of successor)
            new_state = state[i:]  # Next state (successor)
            cost = self.unigramCost(state[:i])  # Cost of current state
            
            list_of_succ_and_cost.append((action, new_state, cost))  # append it in format of util
        
        return list_of_succ_and_cost
        # END_YOUR_CODE


def segmentWords(query: str, unigramCost: Callable[[str], float]) -> str:
    if len(query) == 0:
        return ''

    uniform_cost_search = util.UniformCostSearch()  # Load Uniform Cost Search
    uniform_cost_search.solve(
        problem=WordSegmentationProblem(
            query=query, 
            unigramCost=unigramCost
        )
    )  # Solve

    path = uniform_cost_search.actions  # Get path

    # BEGIN_YOUR_CODE (our solution is 1 lines of code, but don't worry if you deviate from this)
    return " ".join(path)
    # END_YOUR_CODE


############################################################
# Problem 1b: Solve the vowel insertion problem under a bigram cost

class VowelInsertionProblem(util.SearchProblem):
    def __init__(self, queryWords: List[str], bigramCost: Callable[[str, str], float],
            possibleFills: Callable[[str], Set[str]]):
        self.queryWords = queryWords
        self.bigramCost = bigramCost
        self.possibleFills = possibleFills

    def startState(self):
        # BEGIN_YOUR_CODE (our solution is 1 line of code, but don't worry if you deviate from this)
        return (self.queryWords[0], 0)  # state is defined as (word, word_index)
        # END_YOUR_CODE

    def isEnd(self, state) -> bool:
        # BEGIN_YOUR_CODE (our solution is 1 lines of code, but don't worry if you deviate from this)
        return state[1] == (len(self.queryWords) - 1)  # If so, scanning of word are complete
        # END_YOUR_CODE
        
    def succAndCost(self, state):
        # BEGIN_YOUR_CODE (our solution is 4 lines of code, but don't worry if you deviate from this)
        list_of_succ_and_cost = []
        seg, word_index = state  # parse states
        possible_fills = self.possibleFills(self.queryWords[word_index + 1])  # get possible fills
        
        if len(possible_fills) != 0:  # Can insert vowel
            dict_of_words = possible_fills  # insert vowel
        
        else:  # Can't insert vowel
            dict_of_words = {self.queryWords[word_index + 1]}  # move to next word
        
        # Construct successor and cost
        for next_seg in dict_of_words:
            next_state = (next_seg, word_index + 1)  # Next state (successor)
            cost = self.bigramCost(seg, next_seg)  # cost of current state
            list_of_succ_and_cost.append((next_seg, next_state, cost))
            
        return list_of_succ_and_cost
        # END_YOUR_CODE


def insertVowels(queryWords: List[str], bigramCost: Callable[[str, str], float],
        possibleFills: Callable[[str], Set[str]]) -> str:
    # BEGIN_YOUR_CODE (our solution is 3 lines of code, but don't worry if you deviate from this)
    if len(queryWords) == 0:
        return ''
    
    uniform_cost_search = util.UniformCostSearch()  # Load Uniform Cost Search
    uniform_cost_search.solve(
        VowelInsertionProblem(
            queryWords=[wordsegUtil.SENTENCE_BEGIN] + queryWords, 
            bigramCost=bigramCost, 
            possibleFills=possibleFills
        )
    )  # Solve

    path = uniform_cost_search.actions  # Get path
    
    return " ".join(path)
    # END_YOUR_CODE


############################################################
# Problem 1c: Solve the joint segmentation-and-insertion problem

class JointSegmentationInsertionProblem(util.SearchProblem):
    def __init__(self, query: str, bigramCost: Callable[[str, str], float],
            possibleFills: Callable[[str], Set[str]]):
        self.query = query
        self.bigramCost = bigramCost
        self.possibleFills = possibleFills

    def startState(self):
        # BEGIN_YOUR_CODE (our solution is 1 line of code, but don't worry if you deviate from this)
        return (wordsegUtil.SENTENCE_BEGIN, self.query)
        # END_YOUR_CODE

    def isEnd(self, state) -> bool:
        # BEGIN_YOUR_CODE (our solution is 1 lines of code, but don't worry if you deviate from this)
        return len(state[1]) == 0
        # END_YOUR_CODE

    def succAndCost(self, state):
        # BEGIN_YOUR_CODE (our solution is 4 lines of code, but don't worry if you deviate from this)
        seg, query = state
        list_of_succ_and_cost = []
        
        # Generate segment successor
        list_of_spacing_succ = []
        for i in range(len(query), 0, -1):
            list_of_spacing_succ.append((query[:i], query[i:]))
        
        # Generate joint successor
        for front_seg, last_seg in list_of_spacing_succ:
            dict_of_words = self.possibleFills(front_seg)  # Insert vowel
            
            for next_seg in dict_of_words:
                action = next_seg  # Current state (target of succssor)
                next_state = (next_seg, last_seg)  # Next state (successor)
                cost = self.bigramCost(seg, next_seg)  # Cost of current state
                
                list_of_succ_and_cost.append((action, next_state, cost))
        
        return list_of_succ_and_cost
        # END_YOUR_CODE


def segmentAndInsert(query: str, bigramCost: Callable[[str, str], float],
        possibleFills: Callable[[str], Set[str]]) -> str:
    if len(query) == 0:
        return ''

    # BEGIN_YOUR_CODE (our solution is 3 lines of code, but don't worry if you deviate from this)
    uniform_cost_search = util.UniformCostSearch()  # Load Uniform Cost Search
    uniform_cost_search.solve(
        JointSegmentationInsertionProblem(
            query=query,
            bigramCost=bigramCost,
            possibleFills=possibleFills
        )
    )  # Solve
    
    path = uniform_cost_search.actions  # Get Path

    return " ".join(path)
    # END_YOUR_CODE


############################################################
# Problem 2a: Solve the maze search problem with uniform cost search

class MazeProblem(util.SearchProblem):
    def __init__(self, start: tuple, goal: tuple, moveCost: Callable[[tuple, str], float],
            possibleMoves: Callable[[tuple], Set[tuple]]) -> float:
        self.start = start
        self.goal = goal
        self.moveCost = moveCost
        self.possibleMoves = possibleMoves

    def startState(self):
        # BEGIN_YOUR_CODE (our solution is 1 line of code, but don't worry if you deviate from this)
        return self.start
        # END_YOUR_CODE

    def isEnd(self, state) -> bool:
        # BEGIN_YOUR_CODE (our solution is 1 lines of code, but don't worry if you deviate from this)
        return (state[0] == self.goal[0]) and (state[1] == self.goal[1])
        # END_YOUR_CODE

    def succAndCost(self, state):
        # BEGIN_YOUR_CODE (our solution is 2 lines of code, but don't worry if you deviate from this)
        list_of_succ_and_cost = []
        
        # Generate successor and cost
        for possible_move in self.possibleMoves(state):
            action = state  # Current state
            direction, next_state = possible_move  # Parse possible move
            cost = self.moveCost(state, direction)  # Cost of current state to next state

            list_of_succ_and_cost.append((action, next_state, cost))
        
        return list_of_succ_and_cost
        # END_YOUR_CODE
            

def UCSMazeSearch(start: tuple, goal: tuple, moveCost: Callable[[tuple, str], float],
            possibleMoves: Callable[[tuple], Set[tuple]]) -> float:
    ucs = util.UniformCostSearch()
    ucs.solve(MazeProblem(start, goal, moveCost, possibleMoves))
    
    # BEGIN_YOUR_CODE (our solution is 4 lines of code, but don't worry if you deviate from this)
    cost = ucs.totalCost
    return cost
    # END_YOUR_CODE


############################################################
# Problem 2b: Solve the maze search problem with A* search

def consistentHeuristic(goal: tuple):
    def _consistentHeuristic(state: tuple) -> float:
        # BEGIN_YOUR_CODE (our solution is 1 lines of code, but don't worry if you deviate from this)
        return abs(goal[0] - state[0]) + abs(goal[1] - state[1])  # manhattan distance
        # END_YOUR_CODE
    return _consistentHeuristic

def AStarMazeSearch(start: tuple, goal: tuple, moveCost: Callable[[tuple, str], float],
            possibleMoves: Callable[[tuple], Set[tuple]]) -> float:
    ucs = util.UniformCostSearch()
    ucs.solve(MazeProblem(start, goal, moveCost, possibleMoves), heuristic=consistentHeuristic(goal))
    
    # BEGIN_YOUR_CODE (our solution is 13 lines of code, but don't worry if you deviate from this)
    cost = len(ucs.actions)  # cost is one per edge, length of path should be same with total cost 
    return cost
    # END_YOUR_CODE

############################################################


if __name__ == '__main__':
    shell.main()
