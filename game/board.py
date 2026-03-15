WIN_LINES = [
    (0,1,2), (3,4,5), (6,7,8),
    (0,3,6), (1,4,7), (2,5,8),
    (0,4,8), (2,4,6)
]

def check_small_board(board):
    for a, b, c in WIN_LINES:
        s = board[a] + board[b] + board[c]
        if s == 3:
            return 1
        if s == -3:
            return -1
    # No early-termination draw according to Ultimate TTT rules
    if all(cell != 0 for cell in board):
        return 4 # draw
    return 0  # undecided


def get_winning_line(board):
    #Return the indices of the winning line, or None if no win
    for a, b, c in WIN_LINES:
        s = board[a] + board[b] + board[c]
        if s == 3 or s == -3:
            return (a, b, c)
    return None


class UltimateBoard:

    def __init__(self):
        self.boards = [[0]*9 for _ in range(9)]
        self.macro = [0]*9
        self.active_board = None
        self.player = 1  # X starts

    def check_macro(self):
        # check for wins 
        for a, b, c in WIN_LINES:
            line = [self.macro[a], self.macro[b], self.macro[c]]
            if 4 in line:
                continue
            s = sum(line)
            if s == 3:
                return 1
            if s == -3:
                return -1
        
        # check for early-termination draw
        x_can_win = False
        o_can_win = False      
        for a, b, c in WIN_LINES:
            line = [self.macro[a], self.macro[b], self.macro[c]]
            if (-1 not in line) and (4 not in line):
                x_can_win = True
            if (1 not in line) and (4 not in line):
                o_can_win = True      
        if not x_can_win and not o_can_win:
            return 4
        return 0


    def copy(self):
        new = UltimateBoard()
        new.boards = [b[:] for b in self.boards]
        new.macro = self.macro[:]
        new.active_board = self.active_board
        new.player = self.player
        return new
    
    def get_legal_moves(self):
        moves = []

        if self.active_board is None:
            for b in range(9):
                #if self.macro[b] != 0: # does not allow play in empty cells in already won or lost boards
                if self.macro[b] == 4: # allows play in empty cells in already won or lost boards
                    continue
                for c in range(9):
                    if self.boards[b][c] == 0:
                        moves.append((b, c))
        else:
            b = self.active_board
            if self.macro[b] == 0:
                for c in range(9):
                    if self.boards[b][c] == 0:
                        moves.append((b, c))
            else:
                self.active_board = None
                return self.get_legal_moves()

        return moves

    def apply_move(self, move):
        b, c = move

        # Validate move
        if self.boards[b][c] != 0:
            raise ValueError(f"Cell ({b}, {c}) is already occupied")

        self.boards[b][c] = self.player

        # Update small board status
        result = check_small_board(self.boards[b])
        if result != 0:
            self.macro[b] = result

        # Update active board
        self.active_board = c
        if self.macro[c] != 0:
            self.active_board = None

        # Switch player
        self.player *= -1

    def is_terminal(self):
        return self.check_macro() != 0
