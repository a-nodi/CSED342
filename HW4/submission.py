import util, math, random
from collections import defaultdict
from util import ValueIteration
import numpy as np


############################################################
# Problem 1a: Volcano Crossing


class VolcanoCrossing():
    """
    grid_world: a 2D numpy array where 0 is explorable, negative integer is a volcano, and positive integer is the goal.
    discount: discount factor
    moveReward: reward of moving from one cell to another
    value_table: a 2D numpy array where each cell represents the value of the cell
    actions: a list of possible actions
    """
    def __init__(self, grid_world, discount=1, moveReward=-1):
        self.grid_world = grid_world
        self.discount = discount
        self.moveReward = moveReward
        self.actions = [(0, 1), (0, -1), (1, 0), (-1, 0)]

    # Return the value table after running |numIters| of value iteration.
    # You do not need to modify this function.
    def value_iteration(self, numIters=1):
        self.value_table = np.zeros(self.grid_world.shape) # Initialize value table

        for _ in range(numIters):
            self.value_table = self.value_update(self.value_table)
        return self.value_table

    # Return the state is Volcano or Island.
    # You do not need to modify this function.
    # If the state is Volcano or Island, return True.
    # Otherwise(self.grid_world[state] == 0), return False.
    # This function below has been implemented for your convenience, but it is not necessarily required to be used.
    def is_volcano_or_island(self, state):
        return self.grid_world[state] != 0

    # Checks if the agent can move to the next state.
    # This function below has been implemented for your convenience, but it is not necessarily required to be used.
    def movable(self, state, action):
        x, y = state
        i, j = action
        return 0 <= x + i < self.grid_world.shape[0] and 0 <= y + j < self.grid_world.shape[1]


    # Return the value table after updating the value of each grid cell.
    def value_update(self, value_table):
        # BEGIN_YOUR_ANSWER (our solution is 14 lines of code, but don't worry if you deviate from this)
        for i in range(self.grid_world.shape[0]):
            for j in range(self.grid_world.shape[1]):
                
                # Fixed value for Volcano and Island
                if self.is_volcano_or_island((i, j)):
                    value_table[(i, j)] = self.grid_world[(i, j)]
                    continue
                    
                # Initialize max_val
                max_val = float('-inf')
                
                # iterate over all possible actions
                for action in self.actions:
                    # Check if the agent can move to the next state
                    if not self.movable((i, j), action):
                        continue
                    
                    # Calculate the next state
                    x, y = i + action[0], j + action[1]
                    new_state = (x, y)
                    
                    # Update max_val
                    max_val = max(max_val, self.moveReward + self.discount * value_table[new_state])
                
                # Update value_table
                value_table[(i, j)] = max_val
        
        return value_table
        # END_YOUR_ANSWER

############################################################
# Problem 2a: BlackjackMDP


class BlackjackMDP(util.MDP):
    def __init__(self, cardValues, multiplicity, threshold, peekCost):
        """
        cardValues: array of card values for each card type
        multiplicity: number of each card type
        threshold: maximum total before going bust
        peekCost: how much it costs to peek at the next card
        """
        super().__init__()

        self.cardValues = cardValues
        self.multiplicity = multiplicity
        self.threshold = threshold
        self.peekCost = peekCost

    # Return the start state.
    # Look at this function to learn about the state representation.
    # The first element of the tuple is the sum of the cards in the player's
    # hand.
    # The second element is the index (not the value) of the next card, if the player peeked in the
    # last action.  If they didn't peek, this will be None.
    # The final element is the current deck.
    def startState(self):
        return (0, None, (self.multiplicity,) * len(self.cardValues))  # total, next card (if any), multiplicity for each card

    # Return set of actions possible from |state|.
    # You do not need to modify this function.
    # All logic for dealing with end states should be done in succAndProbReward
    def actions(self, state):
        return ['Take', 'Peek', 'Quit']

    # Return a list of (newState, prob, reward) tuples corresponding to edges
    # coming out of |state|.  Indicate a terminal state (after quitting or
    # busting) by setting the deck to None. 
    # When the probability is 0 for a particular transition, don't include that 
    # in the list returned by succAndProbReward.
    def succAndProbReward(self, state, action):
        # BEGIN_YOUR_ANSWER (our solution is 44 lines of code, but don't worry if you deviate from this)
        
        total_card_value_in_hand, next_card_index_if_peeked, deck_card_counts = state
        is_peeked = next_card_index_if_peeked is not None
        
        list_of_succ_prob_reward = []
        
        def get_peeked_deck(_card_index, _deck_card_counts):
            list_of_deck = list(_deck_card_counts)
            list_of_deck[_card_index] -= 1
            
            return tuple(list_of_deck) if sum(list_of_deck) > 0 else None
            
        def peek(_card_index, _deck_card_counts, _total_card_value_in_hand):
            _deck_card_counts = get_peeked_deck(_card_index, _deck_card_counts)
            _value = _total_card_value_in_hand + self.cardValues[_card_index]
            _reward = value if sum(_deck_card_counts) == 0 else 0
            
            return _value, _total_card_value_in_hand, _reward
            
        if deck_card_counts is None:  # END state
            return []
        
        if total_card_value_in_hand > self.threshold:  # Busted
            return []
        
        if action == 'Take':
            if is_peeked:
                card_value = self.cardValues[next_card_index_if_peeked]  # Determine the value of the next card
                is_busted = total_card_value_in_hand + card_value > self.threshold  # Check if the player busts
                
                # Update the deck if the player peeked
                next_deck_card_counts = get_peeked_deck(next_card_index_if_peeked, deck_card_counts)
                next_deck_card_counts = None if is_busted else next_deck_card_counts  # if the player busts, set the deck to None                

                succ = (total_card_value_in_hand + card_value, None, next_deck_card_counts)
                prob = 1
                
                if is_busted:
                    reward = 0
                elif next_deck_card_counts is None:
                    reward = total_card_value_in_hand + card_value
                else:
                    reward = 0
                
                list_of_succ_prob_reward.append((succ, prob, reward))
            
            else:
                total_card_count = sum(deck_card_counts)
                
                for card_index, card_count in enumerate(deck_card_counts):
                    if card_count == 0:
                        continue
                    
                    card_value = self.cardValues[card_index]  # Determine the value of the next card
                    is_busted = total_card_value_in_hand + card_value > self.threshold  # Check if the player busts

                    # Update the deck if the player didn't peek
                    next_deck_card_counts = get_peeked_deck(card_index, deck_card_counts)
                    next_deck_card_counts = None if is_busted else next_deck_card_counts  # if the player busts, set the deck to None                
                    
                    succ = (total_card_value_in_hand + card_value, None, next_deck_card_counts)
                    prob = card_count / total_card_count
                    
                    if is_busted:
                        reward = 0
                    elif next_deck_card_counts is None:
                        reward = total_card_value_in_hand + card_value
                    else:
                        reward = 0
                    
                    list_of_succ_prob_reward.append((succ, prob, reward))
                
        elif action == 'Peek':
            
            total_card_count = sum(deck_card_counts)
            
            # Iterate over all possible card values
            for card_index, card_count in enumerate(deck_card_counts):
                if card_count == 0:
                    continue
                
                # card_value = self.cardValues[card_index]
                succ = (total_card_value_in_hand, card_index, deck_card_counts)
                prob = card_count / total_card_count
                reward = -self.peekCost
                
                list_of_succ_prob_reward.append((succ, prob, reward))
        
        elif action == 'Quit':
            is_busted = total_card_value_in_hand > self.threshold
            succ = (total_card_value_in_hand, None, None)
            prob = 1
            reward = total_card_value_in_hand if not is_busted else 0

            list_of_succ_prob_reward.append((succ, prob, reward))
             
        return list_of_succ_prob_reward
        # END_YOUR_ANSWER

    def discount(self):
        return 1


############################################################
# Problem 3a: Q learning

# Performs Q-learning.  Read util.RLAlgorithm for more information.
# actions: a function that takes a state and returns a list of actions.
# discount: a number between 0 and 1, which determines the discount factor
# featureExtractor: a function that takes a state and action and returns a list of (feature name, feature value) pairs.
# explorationProb: the epsilon value indicating how frequently the policy
# returns a random action
class Qlearning(util.RLAlgorithm):
    def __init__(self, actions, discount, featureExtractor, explorationProb=0.2):
        self.actions = actions
        self.discount = discount
        self.featureExtractor = featureExtractor
        self.explorationProb = explorationProb
        self.weights = defaultdict(float)
        self.numIters = 0

    # Return the Q function associated with the weights and features
    def getQ(self, state, action):
        score = 0
        for f, v in self.featureExtractor(state, action):
            score += self.weights[f] * v
        return score

    # This algorithm will produce an action given a state.
    # Here we use the epsilon-greedy algorithm: with probability
    # |explorationProb|, take a random action.
    def getAction(self, state):
        self.numIters += 1
        if random.random() < self.explorationProb:
            return random.choice(self.actions(state))
        else:
            return max((self.getQ(state, action), action) for action in self.actions(state))[1]

    # Call this function to get the step size to update the weights.
    def getStepSize(self):
        return 1.0 / math.sqrt(self.numIters)

    # We will call this function with episode=[..., state, action,
    # reward, newState], which you should use to update
    # |self.weights|. You should update |self.weights| using
    # self.getStepSize(); use self.getQ() to compute the current
    # estimate of the parameters. Also, you should assume that
    # V_opt(newState)=0 when isLast(newState) is True
    def incorporateFeedback(self, episode, isLast):
        state, action, reward, newState = episode[-4:]

        if isLast(state):
            return

        # BEGIN_YOUR_ANSWER (our solution is 8 lines of code, but don't worry if you deviate from this)
        pred = self.getQ(state, action)
        list_of_q = [self.getQ(newState, newAction) for newAction in self.actions(newState)]
        v_opt = max(list_of_q) if not isLast(newState) else 0
        target = reward + self.discount * v_opt 
        
        # Update weights
        for key, value in self.featureExtractor(state, action):
            self.weights[key] -= self.getStepSize() * (pred - target) * value
        # END_YOUR_ANSWER


############################################################
# Problem 3b: Q SARSA

class SARSA(Qlearning):
    # We will call this function with episode=[..., state, action,
    # reward, newState, newAction, newReward, newNewState], which you
    # should use to update |self.weights|. You should
    # update |self.weights| using self.getStepSize(); use self.getQ()
    # to compute the current estimate of the parameters. Also, you
    # should assume that Q_pi(newState, newAction)=0 when when
    # isLast(newState) is True
    def incorporateFeedback(self, episode, isLast):
        assert (len(episode) - 1) % 3 == 0
        if len(episode) >= 7:
            state, action, reward, newState, newAction = episode[-7: -2]
        else:
            return

        # BEGIN_YOUR_ANSWER (our solution is 8 lines of code, but don't worry if you deviate from this)
        if isLast(state):
            return
        
        ratio = self.getStepSize()
        
        pred = self.getQ(state, action)
        q_pi = self.getQ(newState, newAction) if not isLast(newState) else 0
        target = reward + self.discount * q_pi
        
        for key, value in self.featureExtractor(state, action):
            self.weights[key] -= ratio * (pred - target) * value
        # END_YOUR_ANSWER

# Return a singleton list containing indicator feature (if exist featurevalue = 1)
# for the (state, action) pair.
def identityFeatureExtractor(state, action):
    featureKey = (state, action)
    featureValue = 1
    return [(featureKey, featureValue)]

############################################################
# Problem 3c: features for Q-learning.

# You should return a list of (feature key, feature value) pairs
# (see identityFeatureExtractor() above for an example).
# Implement the following features:
# - indicator on the total and the action (1 feature).
# - indicator on the presence/absence of each card type and the action (1 feature).
#       Example: if the deck is (3, 4, 0 , 2), then your indicator on the presence of each card type is (1,1,0,1)
#       Only add this feature if the deck != None
# - indicator on the number of cards for each card type and the action (len(counts) features).
#       Example: if the deck is (3, 4, 0, 2), you should have four features (one for each card type).
#       And the first feature key will be (0, 3, action)
#       Only add these features if the deck != None

def blackjackFeatureExtractor(state, action):
    total, nextCard, counts = state
    # BEGIN_YOUR_ANSWER (our solution is 8 lines of code, but don't worry if you deviate from this)
    counts = [] if counts is None else counts
    first_feature = ((total, action), 1)
    second_feature = ((tuple([1 if count > 0 else 0 for count in counts]), action), 1)
    third_feature = [((i, count, action), 1) for i, count in enumerate(counts)]
    
    return [first_feature, second_feature, *third_feature]
    
    # END_YOUR_ANSWER
