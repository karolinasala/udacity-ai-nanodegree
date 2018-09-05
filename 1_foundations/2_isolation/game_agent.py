import random


class SearchTimeout(Exception):
    """Subclass base exception for code clarity. """
    pass


def custom_score(game, player):
    """
    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """
    if game.is_loser(player):
        return float("-inf")

    if game.is_winner(player):
        return float("inf")

    player_moves = len(game.get_legal_moves())
    
    return float(player_moves)


def custom_score_2(game, player):
    """
    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    
    player_moves = len(game.get_legal_moves())
    opponent_moves = len(game.get_legal_moves(game.get_opponent(player)))

    return float(player_moves - opponent_moves)
    """
    
    # get current move count
    move_count = game.move_count

    # count number of moves available
    own_moves = len(game.get_legal_moves(player))
    opp_moves = len(game.get_legal_moves(game.get_opponent(player)))

    # calculate weight
    w = 10 / (move_count + 1)

    # return weighted delta of available moves
    return float(own_moves - (w * opp_moves))


def custom_score_3(game, player):
    """Output of this heuristic is equal to the difference
    in number of moves for both players,
    and with the different parameter for opponent.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """

    player_moves = len(game.get_legal_moves())
    opponent_moves = len(game.get_legal_moves(game.get_opponent(player)))

    return float(player_moves - 1.5*opponent_moves)

class IsolationPlayer:
    """Base class for minimax and alphabeta agents -- this class is never
    constructed or tested directly.

    ********************  DO NOT MODIFY THIS CLASS  ********************

    Parameters
    ----------
    search_depth : int (optional)
        A strictly positive integer (i.e., 1, 2, 3,...) for the number of
        layers in the game tree to explore for fixed-depth search. (i.e., a
        depth of one (1) would only explore the immediate sucessors of the
        current state.)

    score_fn : callable (optional)
        A function to use for heuristic evaluation of game states.

    timeout : float (optional)
        Time remaining (in milliseconds) when search is aborted. Should be a
        positive value large enough to allow the function to return before the
        timer expires.
    """
    def __init__(self, search_depth=3, score_fn=custom_score, timeout=10.):
        self.search_depth = search_depth
        self.score = score_fn
        self.time_left = None
        self.TIMER_THRESHOLD = timeout


class MinimaxPlayer(IsolationPlayer):
    """Game-playing agent that chooses a move using depth-limited minimax
    search. You must finish and test this player to make sure it properly uses
    minimax to return a good move before the search time limit expires.
    """

    def get_move(self, game, time_left):
        """Search for the best move from the available legal moves and return a
        result before the time limit expires.

        **************  YOU DO NOT NEED TO MODIFY THIS FUNCTION  *************

        For fixed-depth search, this function simply wraps the call to the
        minimax method, but this method provides a common interface for all
        Isolation agents, and you will replace it in the AlphaBetaPlayer with
        iterative deepening search.

        Parameters
        ----------
        game : `isolation.Board`
            An instance of `isolation.Board` encoding the current state of the
            game (e.g., player locations and blocked cells).

        time_left : callable
            A function that returns the number of milliseconds left in the
            current turn. Returning with any less than 0 ms remaining forfeits
            the game.

        Returns
        -------
        (int, int)
            Board coordinates corresponding to a legal move; may return
            (-1, -1) if there are no available legal moves.
        """
        self.time_left = time_left

        # Initialize the best move so that this function returns something
        # in case the search fails due to timeout
        best_move = (-1, -1)

        try:
            # The try/except block will automatically catch the exception
            # raised when the timer is about to expire.
            return self.minimax(game, self.search_depth)

        except SearchTimeout:
            pass  # Handle any actions required after timeout as needed

        # Return the best move from the last completed search iteration
        return best_move

    def minimax(self, game, depth):
        """Implement depth-limited minimax search algorithm as described in
        the lectures.

        This should be a modified version of MINIMAX-DECISION in the AIMA text.
        https://github.com/aimacode/aima-pseudocode/blob/master/md/Minimax-Decision.md

        **********************************************************************
            You MAY add additional methods to this class, or define helper
                 functions to implement the required functionality.
        **********************************************************************

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        Returns
        -------
        (int, int)
            The board coordinates of the best move found in the current search;
            (-1, -1) if there are no legal moves

        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project tests; you cannot call any other evaluation
                function directly.

            (2) If you use any helper functions (e.g., as shown in the AIMA
                pseudocode) then you must copy the timer check into the top of
                each helper function or else your agent will timeout during
                testing.
        """
        
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()
        
        # Return the result of mini_max function
        return self.mini_max(game, depth)

    def time(self):
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()
    
    #Minimax method. Find the max value.
    def mini_max(self, game, depth):
        #set of valid moves
        moves = game.get_legal_moves()
        if not moves:
            return (-1, -1)
        best_score = float("-inf")
        self.best_move = moves[0]
        #simulates valid moves and evaluates each game state
        for move in moves:
            self.time()
            clone = game.forecast_move(move)
            score = self.min(clone,depth - 1)
            if score > best_score:
                best_score = score
                self.best_move = move
        #return the best move
        return self.best_move

    # maximize player's score
    def max(self,game,depth):
        if depth == 0:
            return self.score(game,self)
        moves = game.get_legal_moves()
        best_score = float("-inf")
        for move in moves:
            self.time()
            clone = game.forecast_move(move)
            score = self.min(clone, depth - 1)
            if score > best_score:
                best_move = move
                best_score = score
        #return the best score
        return best_score

    # opponent minimize player's score 
    def min(self,game,depth):
        if depth == 0:
            return self.score(game,self)
        moves = game.get_legal_moves()
        best_score = float("inf")
        for move in moves:
            self.time()
            clone = game.forecast_move(move)
            score = self.max(clone, depth - 1)
            if score < best_score:
                best_move = move
                best_score = score
        #return the best score
        return best_score 

class AlphaBetaPlayer(IsolationPlayer):
    """Game-playing agent that chooses a move using iterative deepening minimax
    search with alpha-beta pruning. You must finish and test this player to
    make sure it returns a good move before the search time limit expires.
    """

    def get_move(self, game, time_left):
        """Search for the best move from the available legal moves and return a
        result before the time limit expires.

        Modify the get_move() method from the MinimaxPlayer class to implement
        iterative deepening search instead of fixed-depth search.

        **********************************************************************
        NOTE: If time_left() < 0 when this function returns, the agent will
              forfeit the game due to timeout. You must return _before_ the
              timer reaches 0.
        **********************************************************************

        Parameters
        ----------
        game : `isolation.Board`
            An instance of `isolation.Board` encoding the current state of the
            game (e.g., player locations and blocked cells).

        time_left : callable
            A function that returns the number of milliseconds left in the
            current turn. Returning with any less than 0 ms remaining forfeits
            the game.

        Returns
        -------
        (int, int)
            Board coordinates corresponding to a legal move; may return
            (-1, -1) if there are no available legal moves.
        """
        self.time_left = time_left
        move = (-1, -1)
        depth = 1
        try:
            while (True):
                move = self.alphabeta(game, depth)
                depth += 1
        except SearchTimeout:
            return move


    def alphabeta(self, game, depth, alpha=float("-inf"), beta=float("inf")):
        """Implement depth-limited minimax search with alpha-beta pruning as
        described in the lectures.

        This should be a modified version of ALPHA-BETA-SEARCH in the AIMA text
        https://github.com/aimacode/aima-pseudocode/blob/master/md/Alpha-Beta-Search.md

        **********************************************************************
            You MAY add additional methods to this class, or define helper
                 functions to implement the required functionality.
        **********************************************************************

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        alpha : float
            Alpha limits the lower bound of search on minimizing layers

        beta : float
            Beta limits the upper bound of search on maximizing layers

        Returns
        -------
        (int, int)
            The board coordinates of the best move found in the current search;
            (-1, -1) if there are no legal moves

        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project tests; you cannot call any other evaluation
                function directly.

            (2) If you use any helper functions (e.g., as shown in the AIMA
                pseudocode) then you must copy the timer check into the top of
                each helper function or else your agent will timeout during
                testing.
        """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()
        
        # Return the result of alpha_beta function
        return self.alpha_beta(game, depth)

    def time(self):
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()

    # AlphaBeta method
    def alpha_beta(self, game, depth, alpha=float("-inf"), beta=float("inf")):
        moves = game.get_legal_moves()
        if not moves:
            return (-1, -1)
        self.best_move = moves[0]
        for move in moves:
            self.time()
            clone = game.forecast_move(move)
            score = self.min(clone,depth - 1, alpha, beta)
            #choose the higher value
            if score > alpha:
                alpha = score
                self.best_move = move
        #return the best move for player
        return self.best_move

    # maximize score for the player's move
    def max(self, game, depth, alpha, beta):
        if depth == 0:
            return self.score(game,self)
        moves = game.get_legal_moves()
        #simulates value moves
        for move in moves:
            self.time()
            clone = game.forecast_move(move)
            score = self.min(clone, depth - 1, alpha, beta)
            if score > alpha:
                alpha = score
                if alpha >= beta:
                    break
        #return the value of alpha
        return alpha

    # minimize score for the opponent's move
    def min(self, game, depth, alpha, beta):
        if depth == 0:
            return self.score(game,self)
        moves = game.get_legal_moves()
        for move in moves:
            self.time()
            clone = game.forecast_move(move)
            score = self.max(clone, depth - 1, alpha, beta)
            #choose the lower value
            if score < beta:
                beta = score
                if beta <= alpha:
                    break
        #return the value of beta
        return beta
