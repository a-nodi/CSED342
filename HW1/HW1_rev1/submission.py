import collections
import math
from typing import Any, DefaultDict, List, Set, Tuple

############################################################
# Custom Types
# NOTE: You do not need to modify these.

"""
You can think of the keys of the defaultdict as representing the positions in
the sparse vector, while the values represent the elements at those positions.
Any key which is absent from the dict means that that element in the sparse
vector is absent (is zero).
Note that the type of the key used should not affect the algorithm. You can
imagine the keys to be integer indices (e.g., 0, 1, 2) in the sparse vectors,
but it should work the same way with arbitrary keys (e.g., "red", "blue", 
"green").
"""
DenseVector = List[float]
SparseVector = DefaultDict[Any, float]
Position = Tuple[int, int]


############################################################
# Problem 1a
def find_alphabetically_first_word(text: str) -> str:
    """
    Given a string |text|, return the word in |text| that comes first
    lexicographically (i.e., the word that would come first after sorting).
    A word is defined by a maximal sequence of characters without whitespaces.
    You might find min() handy here. If the input text is an empty string,
    it is acceptable to either return an empty string or throw an error.
    """
    # BEGIN_YOUR_CODE (our solution is 1 line of code, but don't worry if you deviate from this)
    return min(text.split())
    # END_YOUR_CODE

############################################################
# Problem 1b
def find_frequent_words(text:str, freq:int)->Set[str]:
    """
    Splits the string |text| by whitespace
    then returns a set of words that appear at a given frequency |freq|.

    For exapmle:
    >>> dense2sparseVector('the quick brown fox jumps over the lazy fox',2)
    # {'the', 'fox'}
    """
    # BEGIN_YOUR_ANSWER (our solution is 3 lines of code, but don't worry if you deviate from this)
    count_of_words = collections.Counter(text.split())
    list_of_freq_words = []
    for key in count_of_words.keys():
        if count_of_words[key] == freq:
            list_of_freq_words.append(key)
    
    return set(list_of_freq_words)
    # END_YOUR_ANSWER 

############################################################
# Problem 1c
def find_nonsingleton_words(text: str) -> Set[str]:
    """
    Split the string |text| by whitespace and return the set of words that
    occur more than once.
    You might find it useful to use collections.defaultdict(int).
    """
    # BEGIN_YOUR_CODE (our solution is 4 lines of code, but don't worry if you deviate from this)
    count_of_words = collections.Counter(text.split())
    list_of_freq_words = []
    for key in count_of_words.keys():
        if count_of_words[key] > 1:
            list_of_freq_words.append(key)
    
    return set(list_of_freq_words)
    # END_YOUR_CODE

############################################################
# Problem 2a
def dense_vector_dot_product(v1:DenseVector, v2:DenseVector)->float:    
    """
    Given two dense vectors |v1| and |v2|, each represented as list,
    return their dot product.
    You might find it useful to use sum(), and zip() and a list comprehension.
    """
    # BEGIN_YOUR_ANSWER (our solution is 1 lines of code, but don't worry if you deviate from this)
    return sum([v1_entry * v2_entry for v1_entry, v2_entry in zip(v1, v2)])
    # END_YOUR_ANSWER

############################################################
# Problem 2b
def increment_dense_vector(v1:DenseVector, scale:float, v2:DenseVector)->DenseVector:
    """
    Given two dense vectors |v1| and |v2| and float scalar value scale, return v = v1 + scale * v2.
    """
    # BEGIN_YOUR_ANSWER (our solution is 1 lines of code, but don't worry if you deviate from this)
    return [v1_entry + scale * v2_entry for v1_entry, v2_entry in zip(v1, v2)]
    # END_YOUR_ANSWER

############################################################
# Problem 2c
def dense_to_sparse_vector(v:DenseVector)->SparseVector:
    """
    Given a dense vector |v|, return its sparse vector form,
    represented as collection.defaultdict(float).
    
    For exapmle:
    >>> dv = [0, 0, 1, 0, 3]
    >>> dense2sparseVector(dv)
    # defaultdict(<class 'float'>, {2: 1, 4: 3})
    
    You might find it useful to use enumerate().
    """
    # BEGIN_YOUR_ANSWER (our solution is 1 lines of code, but don't worry if you deviate from this)
    return {i: entry for i, entry in enumerate(v) if entry != 0}
    # END_YOUR_ANSWER


############################################################
# Problem 2d

def sparse_vector_dot_product(v1: SparseVector, v2: SparseVector) -> float:
    """
    Given two sparse vectors (vectors where most of the elements are zeros)
    |v1| and |v2|, each represented as collections.defaultdict(float), return
    their dot product.

    You might find it useful to use sum() and a list comprehension.
    This function will be useful later for linear classifiers.
    Note: A sparse vector has most of its entries as 0.
    """
    # BEGIN_YOUR_CODE (our solution is 1 line of code, but don't worry if you deviate from this)  
    return sum([v1[key1] * v2[key2] if key1 == key2 else 0 for key1 in v1 for key2 in v2])
    # END_YOUR_CODE


############################################################
# Problem 2e

def increment_sparse_vector(v1: SparseVector, scale: float, v2: SparseVector,
) -> None:
    """
    Given two sparse vectors |v1| and |v2|, perform v1 += scale * v2.
    If the scale is zero, you are allowed to modify v1 to include any
    additional keys in v2, or just not add the new keys at all.

    NOTE: Unlike `increment_dense_vector` , this function should MODIFY v1 in-place, but not return it.
    Do not modify v2 in your implementation.
    This function will be useful later for linear classifiers.
    """
    # BEGIN_YOUR_CODE (our solution is 2 lines of code, but don't worry if you deviate from this)
    for key in set(list(v1.keys()) + list(v2.keys())):
        v1[key] = v1[key] + scale * v2[key]
    # END_YOUR_CODE

############################################################
# Problem 2f

def euclidean_distance(loc1: Position, loc2: Position) -> float:
    """
    Return the Euclidean distance between two locations, where the locations
    are PAIRs of numbers (e.g., (3, 5)).
    NOTE: We only consider 2-dimensional Postions as the parameters of this function. 
    """
    # BEGIN_YOUR_CODE (our solution is 1 line of code, but don't worry if you deviate from this)
    return ((loc1[0] - loc2[0]) ** 2 + (loc1[1] - loc2[1]) ** 2) ** 0.5
    # END_YOUR_CODE

############################################################
# Problem 3a

def mutate_sentences(sentence: str) -> List[str]:
    """
    Given a sentence (sequence of words), return a list of all "similar"
    sentences.
    We define a sentence to be "similar" to the original sentence if
      - it has the same number of words, and
      - each pair of adjacent words in the new sentence also occurs in the
        original sentence (the words within each pair should appear in the same
        order in the output sentence as they did in the original sentence).
    Notes:
      - The order of the sentences you output doesn't matter.
      - You must not output duplicates.
      - Your generated sentence can use a word in the original sentence more
        than once.
    Example:
      - Input: 'the cat and the mouse'
      - Output: ['and the cat and the', 'the cat and the mouse',
                 'the cat and the cat', 'cat and the cat and']
                (Reordered versions of this list are allowed.)
    """
    # BEGIN_YOUR_CODE (our solution is 17 lines of code, but don't worry if you deviate from this)
    list_of_word = sentence.split()
    unique_list_of_word = list(set(list_of_word))
    list_of_sentence = []
    dict_of_word_index = {}
    matrix = [[0 for i in range(len(unique_list_of_word))] for j in range(len(unique_list_of_word))]
    max_path_length = len(list_of_word)
    
    for i, word in enumerate(unique_list_of_word):
        dict_of_word_index[word] = i  
    
    # Construct the adjacency matrix
    for i in range(len(list_of_word) - 1):
        first_word, second_word = list_of_word[i], list_of_word[i + 1]
        matrix[dict_of_word_index[first_word]][dict_of_word_index[second_word]] = 1    
    
    # Construct the sentence using BFS
    for start_word in unique_list_of_word:
        
        queue = []
        
        root = {
            "word": start_word,
            "path": [start_word],
            "next": [word for word in unique_list_of_word if matrix[dict_of_word_index[start_word]][dict_of_word_index[word]] == 1]
        }
        
        queue.append(root)
        
        while queue:
            current_node = queue.pop(0)
            
            if len(current_node["path"]) == max_path_length:
                list_of_sentence.append(' '.join(current_node["path"]))
                continue
                
            for next_word in current_node["next"]:
                next_node = {
                    "word": next_word,
                    "path": current_node["path"] + [next_word],
                    "next": [word for word in unique_list_of_word if matrix[dict_of_word_index[next_word]][dict_of_word_index[word]] == 1]
                }
                queue.append(next_node)
        
    return list(set(list_of_sentence)) 
    # END_YOUR_CODE
