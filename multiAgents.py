# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


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


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {North, South, West, East, Stop}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
        #scared=ghost I can devour
        "*** YOUR CODE HERE ***"
        currScore=scoreEvaluationFunction(currentGameState)                     #current score
        food=[]
        food_coordinates=[]
        scaredghosts=[]                                                         #initialize lists and scared, bad ghosts
        badghosts=[]
        minscared=1000
        minbad=1000
        
        food=newFood.asList()
        
        if successorGameState.isWin():                                          #if next state wins
            return float("inf")
        for i in food:                                                          #append all food positions and keep in closestfood the one closest
            food_coordinates.append(util.manhattanDistance(newPos, i))          #to pacman's position
        closestfood=min(food_coordinates)
        for i in newGhostStates:                                                #for every ghost state
            if newPos==i.getPosition():                                         #if the collide, lose!
                return -1;
            else:
                if i.scaredTimer:                                               #append the scared ghosts 
                    scaredghosts.append(util.manhattanDistance(newPos, i.getPosition()))
                else:                                                           #and the bad ones
                    badghosts.append(util.manhattanDistance(newPos, i.getPosition()))
        if len(scaredghosts):                                                   #if there are any keep the minimum distances from them
            minscared=min(scaredghosts)
        if len(badghosts):                                                      #in any other case, keep the default values
            minbad=min(badghosts)
        score=currScore+ (1.0/closestfood) + (1.0/minscared) - (1.0/minbad) +(1.0/len(food)) + len(currentGameState.getCapsules())  #calculate scores with values that 
        return score+successorGameState.getScore()                              #prove winning(I found that of all the ones I tries, 1.0 works best)
 #return the score and successor's score
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

class MinimaxAgent(MultiAgentSearchAgent):
    """
      Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action from the current gameState using self.depth
          and self.evaluationFunction.

          Here are some method calls that might be useful when implementing minimax.

          gameState.getLegalActions(agentIndex):
            Returns a list of legal actions for an agent
            agentIndex=0 means Pacman, ghosts are >= 1

          gameState.generateSuccessor(agentIndex, action):
            Returns the successor game state after an agent takes an action

          gameState.getNumAgents():
            Returns the total number of agents in the game
        """
        legalactions=gameState.getLegalActions()        #minimax algorithm
        ghosts=gameState.getNumAgents()-1           #all agents minus pacman(agent 0)
        maxaction=[]
        score=-float("inf")                             #initialization
     
        for i in legalactions:                          #for every action allowed
            successor=gameState.generateSuccessor(0,i)
            temp=score                                                              #check for max between children that hold minimum values
            score=max(score, self.minValue(successor,self.depth, 1, ghosts))
            if score>temp:                                                          #store max value
                maxaction.append(i)

        if score==-float("inf"):                                                    #if score not changed return
            return 
        while(len(maxaction)!=0):                                                   #return every item of max values
            return maxaction.pop()

    
    def maxValue(self, gameState, depth, agents):
        if gameState.isWin() or gameState.isLose() or depth==0:                     #base cases
            return self.evaluationFunction(gameState)
        score=-(float("inf"))
        legalactions=gameState.getLegalActions()
        for i in legalactions:                                                      #for every legal move
            successor=gameState.generateSuccessor(0,i)                              #keep max between min values of children
            score=max(score,self.minValue(successor,depth,1,agents))
        return score

    def minValue(self, gamestate, depth, agents, ghosts):
        if gamestate.isWin() or gamestate.isLose() or depth==0:                     #base cases
            return self.evaluationFunction(gamestate)
        score=(float("inf"))
        legalactions=gamestate.getLegalActions(agents)
        if agents==ghosts:                                                          #if only pacman 
            for i in legalactions:
                successor=gamestate.generateSuccessor(agents,i)                     #keep min of max values of children
                score=min(score,self.maxValue(successor,depth-1,ghosts))
        else:                                                                       #if extra ghosts
            for i in legalactions:
                successor=gamestate.generateSuccessor(agents,i)
                score=min(score, self.minValue(successor,depth,agents+1, ghosts))
        return score;

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        legalactions=gameState.getLegalActions()
        ghosts=gameState.getNumAgents()-1
        actions=[]
        score=-(float("inf"))                           #initialize min and max(alpha and beta)
        alpha=-(float("inf"))
        beta=float("inf")

        for i in legalactions:
            successors=gameState.generateSuccessor(0,i)
            temp=score
            score=max(score, self.minab(successors, self.depth, 1, ghosts, alpha, beta))    #keep max of min children values
            if score>temp:
                actions.append(i)                                                           #store max
            if score>beta:
                return score
            alpha=max(alpha, score)
        if score==-(float("inf")):
            return
        while len(actions)!=0:                                                              #return all max actions
            return actions.pop()

    def maxab(self, gameState, depth, ghosts, alpha, beta):     #adding depth and ghosts for recursion
        if gameState.isWin() or gameState.isLose() or depth==0:
            return self.evaluationFunction(gameState)
        v=-(float("inf"))       #v is score, but in Berkeley's picture->v 
        legalactions=gameState.getLegalActions(0)
        for i in legalactions:
            successor=gameState.generateSuccessor(0,i)                                      #Berkeley's picture and Mr.Koubarakis' algorithm
            v=max(v, self.minab(successor, depth, 1, ghosts, alpha, beta))
            if v>beta:
                return v
            alpha=max(alpha, v)
        return v
    def minab(self, gameState, depth, agents, ghosts, alpha, beta):
        v=float("inf")
        legalactions=gameState.getLegalActions(agents)
        if gameState.isWin() or gameState.isLose() or depth==0:
            return self.evaluationFunction(gameState)
        if agents==ghosts:
            for i in legalactions:
                successors=gameState.generateSuccessor(agents, i)
                v=min(v, self.maxab(successors, depth-1, ghosts, alpha, beta))
                if v<alpha:                                                                 #return-start recursion backwards
                    return v
                beta=min(beta, v)
        else:
            for i in legalactions:
                successors=gameState.generateSuccessor(agents, i)
                v=min(v, self.minab(successors, depth, agents+1, ghosts, alpha, beta))
                if v<alpha:
                    break
                beta=min(beta, v)
        return v    

        
class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        "*** YOUR CODE HERE ***"
        legalactions=gameState.getLegalActions()
        ghosts=gameState.getNumAgents()-1
        actions=[]
        score=-(float("inf"))
        for i in legalactions:
            successors=gameState.generateSuccessor(0,i)
            temp=score
            score=max(score, self.expectimin(successors, self.depth, 1, ghosts))        #same thing, with only difference being, expected versus minimum
            if score>temp :
                actions.append(i)
        while len(actions)!=0:
            return actions.pop()
    
    
    def expectimax(self, gameState, depth, ghosts):
        score=-(float("inf"))
        legalactions=gameState.getLegalActions(0)
        if gameState.isWin() or gameState.isLose() or depth==0:
            return self.evaluationFunction(gameState)
        for i in legalactions:
            successors=gameState.generateSuccessor(0,i)
            score=max(score,self.expectimin(successors,depth,1,ghosts))
        return score
    
    def expectimin(self,gameState, depth, agents, ghosts):
        score=0;
        legalactions=gameState.getLegalActions(agents)
        if gameState.isWin() or gameState.isLose() or depth==0:
            return self.evaluationFunction(gameState)
        if agents==ghosts:
            for i in legalactions:
                successors=gameState.generateSuccessor(agents, i)
                score+=self.expectimax(successors,depth-1,ghosts)
                
        else:
            for i in legalactions:
                successors=gameState.generateSuccessor(agents, i)
                score+=self.expectimin(successors, depth, agents+1, ghosts)
        return score

def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    currPos = currentGameState.getPacmanPosition()
    newFood = currentGameState.getFood()
    newGhostStates = currentGameState.getGhostStates()
    currScore=scoreEvaluationFunction(currentGameState)
    food=[]
    f=[]
    food_coordinates=[]
    scaredghosts=[]
    badghosts=[]
    minscared=1000
    minbad=1000
    food=newFood.asList()
    if currentGameState.isWin():                        #base cases
        return float("inf")
    if currentGameState.isLose():
        return -float("inf")
    closestfood=float("inf")
    for i in food:
        f.append(util.manhattanDistance(i, currPos))            
    minfood=min(f)                                      #calculate position of closest food piece
    if minfood<closestfood:
        closestfood=minfood
    food_coordinates.append(closestfood)
    f=food_coordinates.pop()                            #get closest food piecce
    ghosts=currentGameState.getNumAgents()-1
    for i in newGhostStates:                            #for every ghost state
        if currPos==i.getPosition():                    #if eaten in next move return error value
            return -1;
        else:
            if i.scaredTimer:                           #same as Problem 1
                scaredghosts.append(util.manhattanDistance(currPos, i.getPosition()))
            else:
                badghosts.append(util.manhattanDistance(currPos, i.getPosition()))
    if len(scaredghosts):
        minscared=min(scaredghosts)
    if len(badghosts):
        minbad=min(badghosts)                           #calculate score with different coefficients
    endScore=currScore-(0.5*f) - (0.5*minscared)+(5.0*minbad)-(0.5*len(food)) -(0.5*len(currentGameState.getCapsules()))
    return endScore


# Abbreviation
better = betterEvaluationFunction

