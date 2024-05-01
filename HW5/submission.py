from util import manhattanDistance
from game import Directions
import random, util

from game import Agent



class ReflexAgent(Agent):
  """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
  """
  def __init__(self):
    self.lastPositions = []
    self.dc = None

  def getAction(self, gameState):
    """
    getAction chooses among the best options according to the evaluation function.

    getAction takes a GameState and returns some Directions.X for some X in the set {North, South, West, East, Stop}
    ------------------------------------------------------------------------------
    Description of GameState and helper functions:

    A GameState specifies the full game state, including the food, capsules,
    agent configurations and score changes. In this function, the |gameState| argument 
    is an object of GameState class. Following are a few of the helper methods that you 
    can use to query a GameState object to gather information about the present state 
    of Pac-Man, the ghosts and the maze.
    
    gameState.getLegalActions(): 
        Returns the legal actions for the agent specified. Returns Pac-Man's legal moves by default.

    gameState.generateSuccessor(agentIndex, action): 
        Returns the successor state after the specified agent takes the action. 
        Pac-Man is always agent 0.

    gameState.getPacmanState():
        Returns an AgentState object for pacman (in game.py)
        state.configuration.pos gives the current position
        state.direction gives the travel vector

    gameState.getGhostStates():
        Returns list of AgentState objects for the ghosts

    gameState.getNumAgents():
        Returns the total number of agents in the game

    gameState.getScore():
        Returns the score corresponding to the current state of the game
        It corresponds to Utility(s)

    
    The GameState class is defined in pacman.py and you might want to look into that for 
    other helper methods, though you don't need to.
    """
    # Collect legal moves and successor states
    legalMoves = gameState.getLegalActions()

    # Choose one of the best actions
    scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
    bestScore = max(scores)
    bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
    chosenIndex = random.choice(bestIndices) # Pick randomly among the best

    return legalMoves[chosenIndex]

  def evaluationFunction(self, currentGameState, action):
    """
    The evaluation function takes in the current and proposed successor
    GameStates (pacman.py) and returns a number, where higher numbers are better.

    The code below extracts some useful information from the state, like the
    remaining food (oldFood) and Pacman position after moving (newPos).
    newScaredTimes holds the number of moves that each ghost will remain
    scared because of Pacman having eaten a power pellet.
    """
    # Useful information you can extract from a GameState (pacman.py)
    successorGameState = currentGameState.generatePacmanSuccessor(action)
    newPos = successorGameState.getPacmanPosition()
    oldFood = currentGameState.getFood()
    newGhostStates = successorGameState.getGhostStates()
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

    return successorGameState.getScore()

def scoreEvaluationFunction(currentGameState):
  """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
  """
  return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
  """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
  """
  def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
    self.index = 0 # Pacman is always agent index 0
    self.evaluationFunction = util.lookup(evalFn, globals())
    self.depth = int(depth)

######################################################################################
# Problem 1a: implementing minimax

class MinimaxAgent(MultiAgentSearchAgent):
  """
    Your minimax agent (problem 1)
  """

  def getAction(self, gameState):
    """
      Returns the minimax action from the current gameState using self.depth
      and self.evaluationFunction. Terminal states can be found by one of the following: 
      pacman won, pacman lost or there are no legal moves. 

      Here are some method calls that might be useful when implementing minimax.

      gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

      Directions.STOP:
        The stop direction, which is always legal

      gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

      gameState.getNumAgents():
        Returns the total number of agents in the game

      gameState.getScore():
        Returns the score corresponding to the current state of the game
        It corresponds to Utility(s)
    
      gameState.isWin():
        Returns True if it's a winning state
    
      gameState.isLose():
        Returns True if it's a losing state

      self.depth:
        The depth to which search should continue
    """
    # BEGIN_YOUR_ANSWER    
    ACTION = 0
    Q_VALUE = 1    
    PLAYER = 0
    
    def minimaxSearch(gameState, depth, agentIndex):
      if gameState.isWin() or gameState.isLose() or depth == 0:  # Terminal node
        return (None, self.evaluationFunction(gameState))  # No action can be performed, terminal node value
      
      # Iterate Players for each turn (function call)
      nextAgentIndex = (agentIndex + 1) % gameState.getNumAgents()

      # Sub depth if all agents have done their action
      depth = depth - 1 if nextAgentIndex == PLAYER else depth
      
      if agentIndex == PLAYER:  # Current agent is player
        # Recursively search child nodes.
        list_of_action_and_q_value = [(action, minimaxSearch(gameState.generateSuccessor(agentIndex, action), depth, nextAgentIndex)[Q_VALUE]) for action in gameState.getLegalActions(agentIndex)]
        optimal_action, q_value = max(list_of_action_and_q_value, key=lambda pair: pair[Q_VALUE])  # Get player's max action and value
      
      else:  # Current player is ghost
        # Recursively search child nodes.
        list_of_action_and_q_value = [(action, minimaxSearch(gameState.generateSuccessor(agentIndex, action), depth, nextAgentIndex)[Q_VALUE]) for action in gameState.getLegalActions(agentIndex)]
        optimal_action, q_value = min(list_of_action_and_q_value, key=lambda pair: pair[Q_VALUE])  # Get Ghost's min action and value
    
      return (optimal_action, q_value)
    
    return minimaxSearch(gameState, self.depth, PLAYER)[ACTION]  # Search Player's optimal action     
    # END_YOUR_ANSWER
  
  def getQ(self, gameState, action):
    """
      Returns the minimax Q-Value from the current gameState and given action
      using self.depth and self.evaluationFunction.
      Terminal states can be found by one of the following: 
      pacman won, pacman lost or there are no legal moves.
    """
    # BEGIN_YOUR_ANSWER
    ACTION = 0
    Q_VALUE = 1    
    PLAYER = 0
    
    def minimaxSearch(gameState, depth, agentIndex):
      if gameState.isWin() or gameState.isLose() or depth == 0:  # Terminal node
        return (None, self.evaluationFunction(gameState))  # No action can be performed, terminal node Q-value
      
      # Iterate Players for each turn (function call)
      nextAgentIndex = (agentIndex + 1) % gameState.getNumAgents()
      
      # Sub depth if all agents have done their action
      depth = depth - 1 if nextAgentIndex == PLAYER else depth
      
      if agentIndex == PLAYER:  # Current agent is player
        # Recursively search child nodes.
        list_of_action_and_q_value = [(action, minimaxSearch(gameState.generateSuccessor(agentIndex, action), depth, nextAgentIndex)[Q_VALUE]) for action in gameState.getLegalActions(agentIndex)]
        optimal_action, q_value = max(list_of_action_and_q_value, key=lambda pair: pair[Q_VALUE])  # Get player's max action and value
      
      else:  # Current player is ghost
        # Recursively search child nodes.
        list_of_action_and_q_value = [(action, minimaxSearch(gameState.generateSuccessor(agentIndex, action), depth, nextAgentIndex)[Q_VALUE]) for action in gameState.getLegalActions(agentIndex)]
        optimal_action, q_value = min(list_of_action_and_q_value, key=lambda pair: pair[Q_VALUE])  # Get Ghost's min action and value
    
      return (optimal_action, q_value)
    
    return minimaxSearch(gameState.generateSuccessor(PLAYER, action), self.depth, PLAYER + 1)[Q_VALUE]  # Search Q-value of current player's action     
    # END_YOUR_ANSWER

######################################################################################
# Problem 2a: implementing expectimax

class ExpectimaxAgent(MultiAgentSearchAgent):
  """
    Your expectimax agent (problem 2)
  """

  def getAction(self, gameState):
    """
      Returns the expectimax action using self.depth and self.evaluationFunction

      All ghosts should be modeled as choosing uniformly at random from their
      legal moves.
    """

    # BEGIN_YOUR_ANSWER
    ACTION = 0
    Q_VALUE = 1
    PLAYER = 0
    
    def getProb(agentIndex, _action):
      return 1 / len(gameState.getLegalActions(agentIndex))
    
    def expectimaxSearch(gameState, depth, agentIndex):
      if gameState.isWin() or gameState.isLose() or depth == 0:  # Terminal node
        return (None, self.evaluationFunction(gameState))  # No action can be performed, terminal node Q-value
      
      # Iterrate Players for each turn (function call)
      nextAgentIndex = (agentIndex + 1) % gameState.getNumAgents()
      
      # Sub depth if all agents have done their action
      depth = depth - 1 if nextAgentIndex == PLAYER else depth
      
      if agentIndex == PLAYER:  # Current agent is player
        # Recursively search child nodes.
        list_of_action_and_q_value = [(action, expectimaxSearch(gameState.generateSuccessor(agentIndex, action), depth, nextAgentIndex)[Q_VALUE]) for action in gameState.getLegalActions(agentIndex)]
        optimal_action, q_value = max(list_of_action_and_q_value, key=lambda pair: pair[Q_VALUE])
        
      else:  # Current agent is ghosts
        # Recurisvely search child nodes.
        list_of_q_value = [expectimaxSearch(gameState.generateSuccessor(agentIndex, action), depth, nextAgentIndex)[Q_VALUE] for action in gameState.getLegalActions(agentIndex)]
        list_of_prob = [getProb(agentIndex, action) for action in gameState.getLegalActions(agentIndex)]
        optimal_action, q_value = None, sum([prob * partial_q for prob, partial_q in zip(list_of_prob, list_of_q_value)])
      
      return (optimal_action, q_value)
    
    return expectimaxSearch(gameState, self.depth, PLAYER)[ACTION]
    # END_YOUR_ANSWER
  
  def getQ(self, gameState, action):
    """
      Returns the expectimax Q-Value using self.depth and self.evaluationFunction.
    """
    # BEGIN_YOUR_ANSWER
    ACTION = 0
    Q_VALUE = 1
    PLAYER = 0
    
    def getProb(agentIndex, _action):
      return 1 / len(gameState.getLegalActions(agentIndex))
    
    def expectimaxSearch(gameState, depth, agentIndex):
      if gameState.isWin() or gameState.isLose() or depth == 0:  # Terminal node
        return (None, self.evaluationFunction(gameState))  # No action can be performed, terminal node Q-value
      
      # Iterrate Players for each turn (function call)
      nextAgentIndex = (agentIndex + 1) % gameState.getNumAgents()
      
      # Sub depth if all agents have done their action
      depth = depth - 1 if nextAgentIndex == PLAYER else depth
      
      if agentIndex == PLAYER:  # Current agent is player
        # Recursively search child nodes.
        list_of_action_and_q_value = [(action, expectimaxSearch(gameState.generateSuccessor(agentIndex, action), depth, nextAgentIndex)[Q_VALUE]) for action in gameState.getLegalActions(agentIndex)]
        optimal_action, q_value = max(list_of_action_and_q_value, key=lambda pair: pair[Q_VALUE])
        
      else:  # Current agent is ghosts
        # Recurisvely search child nodes.
        list_of_q_value = [expectimaxSearch(gameState.generateSuccessor(agentIndex, action), depth, nextAgentIndex)[Q_VALUE] for action in gameState.getLegalActions(agentIndex)]
        list_of_prob = [getProb(agentIndex, action) for action in gameState.getLegalActions(agentIndex)]
        optimal_action, q_value = None, sum([prob * partial_q for prob, partial_q in zip(list_of_prob, list_of_q_value)])
      
      return (optimal_action, q_value)
    
    return expectimaxSearch(gameState.generateSuccessor(PLAYER, action), self.depth, PLAYER + 1)[Q_VALUE]
    
    # END_YOUR_ANSWER

######################################################################################
# Problem 3a: implementing biased-expectimax

class BiasedExpectimaxAgent(MultiAgentSearchAgent):
  """
    Your biased-expectimax agent (problem 3)
  """

  def getAction(self, gameState):
    """
      Returns the biased-expectimax action using self.depth and self.evaluationFunction

      All ghosts should be modeled as choosing stop-biasedly from their
      legal moves.
    """

    # BEGIN_YOUR_ANSWER
    ACTION = 0
    Q_VALUE = 1
    PLAYER = 0
    
    def getProb(agentIndex, _action):
      return 0.5 + 0.5 / len(gameState.getLegalActions(agentIndex)) if _action == Directions.STOP else 0.5 / len(gameState.getLegalActions(agentIndex))
    
    def biasedExpectimaxSearch(gameState, depth, agentIndex):
      if gameState.isWin() or gameState.isLose() or depth == 0:  # Terminal node
        return (None, self.evaluationFunction(gameState))  # No action can be performed, terminal node Q-value
      
      # Iterrate Players for each turn (function call)
      nextAgentIndex = (agentIndex + 1) % gameState.getNumAgents()
      
      # Sub depth if all agents have done their action
      depth = depth - 1 if nextAgentIndex == PLAYER else depth
      
      if agentIndex == PLAYER:  # Current agent is player
        # Recursively search child nodes.
        list_of_action_and_q_value = [(action, biasedExpectimaxSearch(gameState.generateSuccessor(agentIndex, action), depth, nextAgentIndex)[Q_VALUE]) for action in gameState.getLegalActions(agentIndex)]
        optimal_action, q_value = max(list_of_action_and_q_value, key=lambda pair: pair[Q_VALUE])
        
      else:  # Current agent is ghosts
        # Recurisvely search child nodes.
        list_of_q_value = [biasedExpectimaxSearch(gameState.generateSuccessor(agentIndex, action), depth, nextAgentIndex)[Q_VALUE] for action in gameState.getLegalActions(agentIndex)]
        list_of_prob = [getProb(agentIndex, action) for action in gameState.getLegalActions(agentIndex)]
        optimal_action, q_value = None, sum([prob * partial_q for prob, partial_q in zip(list_of_prob, list_of_q_value)])
      
      return (optimal_action, q_value)
    
    return biasedExpectimaxSearch(gameState, self.depth, PLAYER)[ACTION]
    # END_YOUR_ANSWER
  
  def getQ(self, gameState, action):
    """
      Returns the biased-expectimax Q-Value using self.depth and self.evaluationFunction.
    """
    # BEGIN_YOUR_ANSWER
    ACTION = 0
    Q_VALUE = 1
    PLAYER = 0
    
    def getProb(agentIndex, _action):
      return 0.5 + 0.5 / len(gameState.getLegalActions(agentIndex)) if _action == Directions.STOP else 0.5 / len(gameState.getLegalActions(agentIndex))
    
    def biasedExpectimaxSearch(gameState, depth, agentIndex):
      if gameState.isWin() or gameState.isLose() or depth == 0:  # Terminal node
        return (None, self.evaluationFunction(gameState))  # No action can be performed, terminal node Q-value
      
      # Iterrate Players for each turn (function call)
      nextAgentIndex = (agentIndex + 1) % gameState.getNumAgents()
      
      # Sub depth if all agents have done their action
      depth = depth - 1 if nextAgentIndex == PLAYER else depth
      
      if agentIndex == PLAYER:  # Current agent is player
        # Recursively search child nodes.
        list_of_action_and_q_value = [(action, biasedExpectimaxSearch(gameState.generateSuccessor(agentIndex, action), depth, nextAgentIndex)[Q_VALUE]) for action in gameState.getLegalActions(agentIndex)]
        optimal_action, q_value = max(list_of_action_and_q_value, key=lambda pair: pair[Q_VALUE])
        
      else:  # Current agent is ghosts
        # Recurisvely search child nodes.
        list_of_q_value = [biasedExpectimaxSearch(gameState.generateSuccessor(agentIndex, action), depth, nextAgentIndex)[Q_VALUE] for action in gameState.getLegalActions(agentIndex)]
        list_of_prob = [getProb(agentIndex, action) for action in gameState.getLegalActions(agentIndex)]
        optimal_action, q_value = None, sum([prob * partial_q for prob, partial_q in zip(list_of_prob, list_of_q_value)])
      
      return (optimal_action, q_value)
    
    return biasedExpectimaxSearch(gameState.generateSuccessor(PLAYER, action), self.depth, PLAYER + 1)[Q_VALUE]
    # END_YOUR_ANSWER

######################################################################################
# Problem 4a: implementing expectiminimax

class ExpectiminimaxAgent(MultiAgentSearchAgent):
  """
    Your expectiminimax agent (problem 4)
  """

  def getAction(self, gameState):
    """
      Returns the expectiminimax action using self.depth and self.evaluationFunction

      The even-numbered ghost should be modeled as choosing uniformly at random from their
      legal moves.
    """

    # BEGIN_YOUR_ANSWER
    ACTION = 0
    Q_VALUE = 1
    PLAYER = 0
    
    def getProb(agentIndex, _action):
      return 1 / len(gameState.getLegalActions(agentIndex))
    
    def expectiminimaxSearch(gameState, depth, agentIndex):
      if gameState.isWin() or gameState.isLose() or depth == 0:  # Terminal node
        return (None, self.evaluationFunction(gameState))  # No action can be performed, terminal node Q-value
      
      # Iterrate Players for each turn (function call)
      nextAgentIndex = (agentIndex + 1) % gameState.getNumAgents()
      
      # Sub depth if all agents have done their action
      depth = depth - 1 if nextAgentIndex == PLAYER else depth
      
      if agentIndex == PLAYER:  # Current agent is player
        # Recursively search child nodes.
        list_of_action_and_q_value = [(action, expectiminimaxSearch(gameState.generateSuccessor(agentIndex, action), depth, nextAgentIndex)[Q_VALUE]) for action in gameState.getLegalActions(agentIndex)]
        optimal_action, q_value = max(list_of_action_and_q_value, key=lambda pair: pair[Q_VALUE])
        
      elif agentIndex % 2 == 1:  # Current agent is odd-numbered ghosts
        # Recursively search child nodes.
        list_of_action_and_q_value = [(action, expectiminimaxSearch(gameState.generateSuccessor(agentIndex, action), depth, nextAgentIndex)[Q_VALUE]) for action in gameState.getLegalActions(agentIndex)]
        optimal_action, q_value = min(list_of_action_and_q_value, key=lambda pair: pair[Q_VALUE])  # Get Ghost's min action and value
        
      elif agentIndex % 2 == 0:  # Current agent is even-numbered ghosts
        # Recurisvely search child nodes.
        list_of_q_value = [expectiminimaxSearch(gameState.generateSuccessor(agentIndex, action), depth, nextAgentIndex)[Q_VALUE] for action in gameState.getLegalActions(agentIndex)]
        list_of_prob = [getProb(agentIndex, action) for action in gameState.getLegalActions(agentIndex)]
        optimal_action, q_value = None, sum([prob * partial_q for prob, partial_q in zip(list_of_prob, list_of_q_value)])
      
      return (optimal_action, q_value)
    
    return expectiminimaxSearch(gameState, self.depth, PLAYER)[ACTION]
    # END_YOUR_ANSWER
  
  def getQ(self, gameState, action):
    """
      Returns the expectiminimax Q-Value using self.depth and self.evaluationFunction.
    """
    # BEGIN_YOUR_ANSWER
    ACTION = 0
    Q_VALUE = 1
    PLAYER = 0
    
    def getProb(agentIndex, _action):
      return 1 / len(gameState.getLegalActions(agentIndex))
    
    def expectiminimaxSearch(gameState, depth, agentIndex):
      if gameState.isWin() or gameState.isLose() or depth == 0:  # Terminal node
        return (None, self.evaluationFunction(gameState))  # No action can be performed, terminal node Q-value
      
      # Iterrate Players for each turn (function call)
      nextAgentIndex = (agentIndex + 1) % gameState.getNumAgents()
      
      # Sub depth if all agents have done their action
      depth = depth - 1 if nextAgentIndex == PLAYER else depth
      
      if agentIndex == PLAYER:  # Current agent is player
        # Recursively search child nodes.
        list_of_action_and_q_value = [(action, expectiminimaxSearch(gameState.generateSuccessor(agentIndex, action), depth, nextAgentIndex)[Q_VALUE]) for action in gameState.getLegalActions(agentIndex)]
        optimal_action, q_value = max(list_of_action_and_q_value, key=lambda pair: pair[Q_VALUE])
        
      elif agentIndex % 2 == 1:  # Current agent is odd-numbered ghosts
        # Recursively search child nodes.
        list_of_action_and_q_value = [(action, expectiminimaxSearch(gameState.generateSuccessor(agentIndex, action), depth, nextAgentIndex)[Q_VALUE]) for action in gameState.getLegalActions(agentIndex)]
        optimal_action, q_value = min(list_of_action_and_q_value, key=lambda pair: pair[Q_VALUE])  # Get Ghost's min action and value
        
      elif agentIndex % 2 == 0:  # Current agent is even-numbered ghosts
        # Recurisvely search child nodes.
        list_of_q_value = [expectiminimaxSearch(gameState.generateSuccessor(agentIndex, action), depth, nextAgentIndex)[Q_VALUE] for action in gameState.getLegalActions(agentIndex)]
        list_of_prob = [getProb(agentIndex, action) for action in gameState.getLegalActions(agentIndex)]
        optimal_action, q_value = None, sum([prob * partial_q for prob, partial_q in zip(list_of_prob, list_of_q_value)])
      
      return (optimal_action, q_value)
    
    return expectiminimaxSearch(gameState.generateSuccessor(PLAYER, action), self.depth, PLAYER + 1)[Q_VALUE]
    # END_YOUR_ANSWER

######################################################################################
# Problem 5a: implementing alpha-beta

class AlphaBetaAgent(MultiAgentSearchAgent):
  """
    Your expectiminimax agent with alpha-beta pruning (problem 5)
  """

  def getAction(self, gameState):
    """
      Returns the expectiminimax action using self.depth and self.evaluationFunction
    """

    # BEGIN_YOUR_ANSWER
    ACTION = 0
    Q_VALUE = 1
    PLAYER = 0
    
    def getProb(agentIndex, _action):
      return 1 / len(gameState.getLegalActions(agentIndex))
    
    def alphabetaSearch(gameState, depth, agentIndex, alpha, beta):
      if gameState.isWin() or gameState.isLose() or depth == 0:  # Terminal node
        return (None, self.evaluationFunction(gameState))  # No action can be performed, terminal node Q-value
      
      # Iterrate Players for each turn (function call)
      nextAgentIndex = (agentIndex + 1) % gameState.getNumAgents()
      
      # Sub depth if all agents have done their action
      depth = depth - 1 if nextAgentIndex == PLAYER else depth
      
      if agentIndex == PLAYER:  # Current agent is player
        optimal_action, q_value = None, float('-inf')
        # Recursively search child nodes.
        for action in gameState.getLegalActions(agentIndex):
          _action, _q_value = alphabetaSearch(gameState.generateSuccessor(agentIndex, action), depth, nextAgentIndex, alpha, beta)
          optimal_action, q_value = max((action, _q_value), (optimal_action, q_value), key=lambda pair: pair[Q_VALUE])
          alpha = max(alpha, q_value)  # Update alpha 
          
          if beta <= alpha:  # Pruning
            break
        
      elif agentIndex % 2 == 1:  # Current agent is odd-numbered ghosts
        optimal_action, q_value = None, float('inf')
        # Recursively search child nodes.
        for action in gameState.getLegalActions(agentIndex):
          _action, _q_value = alphabetaSearch(gameState.generateSuccessor(agentIndex, action), depth, nextAgentIndex, alpha, beta)
          optimal_action, q_value = min((action, _q_value), (optimal_action, q_value), key=lambda pair: pair[Q_VALUE])
          beta = min(beta, q_value)  # Update beta
        
          if beta <= alpha:  # Pruning
            break
        
      elif agentIndex % 2 == 0:  # Current agent is even-numbered ghosts
        # Recurisvely search child nodes.
        list_of_q_value = [alphabetaSearch(gameState.generateSuccessor(agentIndex, action), depth, nextAgentIndex, alpha, beta)[Q_VALUE] for action in gameState.getLegalActions(agentIndex)]
        list_of_prob = [getProb(agentIndex, action) for action in gameState.getLegalActions(agentIndex)]
        optimal_action, q_value = None, sum([prob * partial_q for prob, partial_q in zip(list_of_prob, list_of_q_value)])
      
      return (optimal_action, q_value)
    
    return alphabetaSearch(gameState, self.depth, PLAYER, float('-inf'), float('inf'))[ACTION]
    # END_YOUR_ANSWER
  
  def getQ(self, gameState, action):
    """
      Returns the expectiminimax Q-Value using self.depth and self.evaluationFunction.
    """
    # BEGIN_YOUR_ANSWER
    ACTION = 0
    Q_VALUE = 1
    PLAYER = 0
    
    def getProb(agentIndex, _action):
      return 1 / len(gameState.getLegalActions(agentIndex))
    
    def alphabetaSearch(gameState, depth, agentIndex, alpha, beta):
      if gameState.isWin() or gameState.isLose() or depth == 0:  # Terminal node
        return (None, self.evaluationFunction(gameState))  # No action can be performed, terminal node Q-value
      
      # Iterrate Players for each turn (function call)
      nextAgentIndex = (agentIndex + 1) % gameState.getNumAgents()
      
      # Sub depth if all agents have done their action
      depth = depth - 1 if nextAgentIndex == PLAYER else depth
      
      if agentIndex == PLAYER:  # Current agent is player
        optimal_action, q_value = None, float('-inf')
        # Recursively search child nodes.
        for action in gameState.getLegalActions(agentIndex):
          _action, _q_value = alphabetaSearch(gameState.generateSuccessor(agentIndex, action), depth, nextAgentIndex, alpha, beta)
          optimal_action, q_value = (action, _q_value) if _q_value > q_value else (optimal_action, q_value)
          alpha = max(alpha, q_value)  # Update alpha
          
          if beta <= alpha:  # Pruning
            break
        
      elif agentIndex % 2 == 1:  # Current agent is odd-numbered ghosts
        optimal_action, q_value = None, float('inf')
        # Recursively search child nodes.
        for action in gameState.getLegalActions(agentIndex):
          _action, _q_value = alphabetaSearch(gameState.generateSuccessor(agentIndex, action), depth, nextAgentIndex, alpha, beta)
          optimal_action, q_value = (action, _q_value) if _q_value < q_value else (optimal_action, q_value)
          beta = min(beta, q_value)  # Update beta
        
          if beta <= alpha:  # Pruning
            break
        
      elif agentIndex % 2 == 0:  # Current agent is even-numbered ghosts
        # Recurisvely search child nodes.
        list_of_q_value = [alphabetaSearch(gameState.generateSuccessor(agentIndex, action), depth, nextAgentIndex, alpha, beta)[Q_VALUE] for action in gameState.getLegalActions(agentIndex)]
        list_of_prob = [getProb(agentIndex, action) for action in gameState.getLegalActions(agentIndex)]
        optimal_action, q_value = None, sum([prob * partial_q for prob, partial_q in zip(list_of_prob, list_of_q_value)])
      
      return (optimal_action, q_value)
    
    return alphabetaSearch(gameState.generateSuccessor(PLAYER, action), self.depth, PLAYER + 1, float('-inf'), float('inf'))[Q_VALUE]
    # END_YOUR_ANSWER

######################################################################################
# Problem 6a: creating a better evaluation function

def betterEvaluationFunction(currentGameState):
  """
  Your extreme, unstoppable evaluation function (problem 6).
  """

  # BEGIN_YOUR_ANSWER

  """
    features can be used for evaluation function
     1. ghost_distance: distance between pacman and ghost
     2. food_distance: distance between pacman and food
     3. capsule_distance: distance between pacman and capsule
    
     4. food_count: number of food
     5. capsule_count: number of capsule
     6. scared_ghost_count: number of scared ghost
     7. ghost_count: number of ghost
    
     8. closest_food_distance: distance between pacman and closest food
     9. closest_capsule_distance: distance between pacman and closest
    10. closest_scared_ghost_distance: distance between pacman and closest scared ghost
    11. closest_ghost_distance: distance between pacman and closest ghost
    
  """

  dict_of_weights = {
    "food_distance": -1,
    "capsule_distance": -1,
    "scared_ghost_distance": 1,
    "ghost_distance": -1,
    
    "food_count": -1,
    "capsule_count": -1,
    "scared_ghost_count": 1,
    "ghost_count": -1,
    
    "closest_food_distance": -1,
    "closest_capsule_distance": -1,
    "closest_scared_ghost_distance": 1,
    "closest_ghost_distance": -1,
  }
  
  position_of_pacman = currentGameState.getPacmanPosition()
  position_of_ghosts = currentGameState.getGhostPositions()
  position_of_food = currentGameState.getFood().asList()
  position_of_capsules = currentGameState.getCapsules()
  
  
  food_distance = []
  
  dict_of_features = {
    "sum_of_food_distance": 0,
    "sum_of_capsule_distance": 0,
    "sum_of_scared_ghost_distance": 0,
    "sum_of_ghost_distance": 0,
    
    "food_count": 0,
    "capsule_count": 0,
    "scared_ghost_count": 0,
    "ghost_count": 0,
    
    "closest_food_distance": 0,
    "closest_capsule_distance": 0,
    "closest_scared_ghost_distance": 0,
    "closest_ghost_distance": 0,
  }
  
  score = 0
  
  for f, w in dict_of_weights.items():
    score += w * dict_of_features[f]
  
  return score
  # END_YOUR_ANSWER

def choiceAgent():
  """
    Choose the pacman agent model you want for problem 6.
    You can choose among the agents above or design your own agent model.
    You should return the name of class of pacman agent.
    (e.g. 'MinimaxAgent', 'BiasedExpectimaxAgent', 'MyOwnAgent', ...)
  """
  # BEGIN_YOUR_ANSWER
  option = 5 - 1  # Index of the agent will be choosed 
  
  list_of_agent_names = [
    'MinimaxAgent',
    'ExpectimaxAgent',
    'BiasedExpectimaxAgent',
    'ExpectiminimaxAgent',
    'AlphaBetaAgent'
  ]
  
  return list_of_agent_names[option]
  # END_YOUR_ANSWER

# Abbreviation
better = betterEvaluationFunction
