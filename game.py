import time
import argparse
import numpy as np


class Game:
    def __init__(self, m, n, k, prune=False):
        """
        Initialize the game parameters.

        Args:
            m (int): Number of columns in the board
            n (int): Number of rows in the board
            k (int): Number of connected pieces needed to win
            prune (bool): Boolean flag to enable or disable alpha-beta pruning
        """
        self.m = m
        self.n = n
        self.k = k
        self.prune = prune
        self.current_player = "Max"
        self.directions = [
            (1, 0),  # vertical
            (0, 1),  # horizontal
            (1, 1),  # main diagonal
            (1, -1), # anti diagonal
        ]
        self.visited = 0
        self.initialize_board()

    def initialize_board(self):
        """Initialize the empty board."""
        self.board = np.full((self.n, self.m), " ")

    def drawboard(self):
        """Draw the current state of the board"""
        print(" " + "--- " * self.m)
        for row in self.board.tolist():
            print("| " + " | ".join(row) + " |")
        print(" " + "--- " * self.m)

    def get_states(self):
        """
        Get all possible (row, col) pairs where a piece can be dropped.

        Returns:
            list[tuple]: (row, col) where a piece can be dropped
        """
        states = []
        for col in range(self.m):
            row = np.where(self.board[:, col] == " ")[0]
            if len(row) == 0:
                continue
            states.append((row[-1], col))
        return states

    def drop_piece(self, column, piece):
        """
        Drop a piece into the specified column.

        Args:
            column (int): Column to drop the piece
            piece (char): Piece to drop (X or O)
        """
        row = np.where(self.board[:, column] == " ")[0][-1]
        self.board[row, column] = piece

    def is_terminal(self):
        """Checks if the game has reached a terminal state

        Returns:
            bool: Whether the current state of the game is terminal
            str: Winner of the game. Possible values - X, O, Draw, None
        """
        rows, cols = np.where(self.board != " ")
        for row, col in zip(rows, cols):
            if self.check_winner(row, col):
                return True, self.board[row, col]

        if " " not in self.board:
            return True, "Draw"
        else:
            return False, None

    def check_winner(self, row, col):
        """Check if there is a winning sequence starting at (row, col)

        Args:
            row (int): Row to check for winning sequence
            col (int): Column to check for winning sequence

        Returns:
            bool: Whether there is a winning sequence
        """
        for dr, dc in self.directions:
            count = 1

            for i in range(1, self.k):
                r, c = row + dr * i, col + dc * i
                if (
                    0 <= r < self.n
                    and 0 <= c < self.m
                    and self.board[r, c] == self.board[row, col]
                ):
                    count += 1
                else:
                    break

            if count == self.k:
                return True

        return False

    def strategy_for_max(self, alpha, beta):
        """Compute the Minimax value for Max, with optional alpha-beta pruning

        Args:
            alpha (float): alpha for alpha-beta pruning. Used only if self.prune is True
            beta (float): beta for alpha-beta pruning. Used only if self.prune is True

        Returns:
            int: Utility of current state. Possible values -1, 0, 1
            int: Best column to choose for Max. Ranges from [0, self.m)
        """
        is_term, winner = self.is_terminal()
        if is_term:
            if winner == "X":
                return (1,)
            elif winner == "O":
                return (-1,)
            else:
                return (0,)

        value = -float("inf")
        best_move = None
        for row, col in self.get_states():
            self.visited += 1
            self.board[row, col] = "X"
            new_value = max(value, self.strategy_for_min(alpha, beta)[0])
            if new_value > value:
                value = new_value
                best_move = col
            self.board[row, col] = " "

            if self.prune:
                if new_value >= beta:
                    return new_value, best_move
                alpha = max(alpha, new_value)

        return value, best_move

    def strategy_for_min(self, alpha, beta):
        """Compute the Minimax value for Min, with optional alpha-beta pruning

        Args:
            alpha (float): alpha for alpha-beta pruning. Used only if self.prune is True
            beta (float): beta for alpha-beta pruning. Used only if self.prune is True

        Returns:
            int: Utility of current state. Possible values -1, 0, 1
            int: Best column to choose for Min. Ranges from [0, self.m)
        """
        is_term, winner = self.is_terminal()
        if is_term:
            if winner == "X":
                return (1,)
            elif winner == "O":
                return (-1,)
            else:
                return (0,)

        value = float("inf")
        best_move = None
        for row, col in self.get_states():
            self.visited += 1
            self.board[row, col] = "O"
            new_value = min(value, self.strategy_for_max(alpha, beta)[0])
            if new_value < value:
                value = new_value
                best_move = col
            self.board[row, col] = " "

            if self.prune:
                if new_value <= alpha:
                    return new_value, best_move
                beta = min(beta, new_value)

        return value, best_move

    def play(self):
        """Main game play loop. User plays as Max, with recommended moves from Minimax algorithm"""
        alpha = -float("inf")
        beta = float("inf")
        while True:
            self.drawboard()
            is_term, winner = self.is_terminal()
            if is_term:
                if winner == "Draw":
                    print("It's a draw!")
                else:
                    print(f"{winner} wins!")
                break

            if self.current_player == "Max":
                print("Max's turn (X):")
                value, move = self.strategy_for_max(alpha, beta)
                print(f"Recommended move: {move}")
                column = int(input("Choose a column: "))
                while not 0 <= column < self.m:
                    column = int(input(f"Invalid input. Choose a column: "))

                self.drop_piece(column, "X")
                self.current_player = "Min"
            else:
                print("Min's turn (O):")
                value, move = self.strategy_for_min(alpha, beta)
                self.drop_piece(move, "O")
                self.current_player = "Max"

    def benchmark_play(self):
        """
        Automatic game loop, used for benchmarking run-time calculations.
        Moves recommended by Minimax algorithm are chosen to play the game.
        """
        alpha = -float("inf")
        beta = float("inf")
        action_times = []
        t1 = time.time()
        while True:
            is_term, winner = self.is_terminal()
            if is_term:
                break

            t11 = time.time()
            if self.current_player == "Max":
                value, move = self.strategy_for_max(alpha, beta)
                self.drop_piece(move, "X")
                self.current_player = "Min"
            else:
                value, move = self.strategy_for_min(alpha, beta)
                self.drop_piece(move, "O")
                self.current_player = "Max"
            t12 = time.time()
            action_times.append(t12 - t11)

        t2 = time.time()
        print(f"Total Runtime: {t2 - t1:.6}s")
        print(f"First Action Time: {action_times[0]:.6}s")
        print(f"Number of visited states: {self.visited}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser() # Importing argparse to handle command-line arguments

    # Adding command-line arguments to customize game configuration

    parser.add_argument("-m", "--m", type=int, default=3, help="number of columns")
    parser.add_argument("-n", "--n", type=int, default=3, help="number of rows")
    parser.add_argument("-k", "--k", type=int, default=3, help="number of connected pieces needed to win")

    # Option to enable alpha-beta pruning for AI optimization

    parser.add_argument("--prune", action="store_true", help="use alpha-beta pruning")

    # Option to enable alpha-beta pruning for AI optimization

    parser.add_argument("--benchmark", action="store_true", help="run performance benchmarks")
    args = parser.parse_args()

    game = Game(args.m, args.n, args.k, args.prune) # Creating a new game instance with the provided arguments
    if args.benchmark:
        game.benchmark_play()  # If benchmark mode is enabled, run performance tests
    else:
        game.play() # Otherwise, start the interactive game
