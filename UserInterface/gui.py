import tkinter as tk
from tkinter import font as tkfont
from game.board import UltimateBoard, get_winning_line

# ─── Layout Constants ───
CELL = 56
PAD = 6          # gap between mini-boards
BOARD = CELL * 3
MACRO = BOARD * 3 + PAD * 2
HEADER_H = 48
CANVAS_W = MACRO + 2 * PAD
CANVAS_H = MACRO + 2 * PAD

# ─── Color Palette (dark theme) ───
BG          = "#0f1117"
SURFACE     = "#181a22"
SURFACE2    = "#22252f"
GRID        = "#2a2d3a"
GRID_MAJOR  = "#363a4a"
TEXT        = "#e8e9ed"
TEXT_DIM    = "#7a7e8f"
TEXT_MUTED  = "#4a4e5f"
X_COLOR     = "#6ee7b7"
O_COLOR     = "#f9a8d4"
LEGAL_FILL  = "#29271e"
LEGAL_BORDER= "#7b6a1e"
LAST_MOVE   = "#fbbf24"
WON_X_BG    = "#112921"
WON_O_BG    = "#251520"
DRAW_BG     = "#1a1c24"

AGENT_DELAY_MS = 500  # Default delay between agent moves (configurable)


class UltimateTTTGUI:
    def __init__(self, board, agent_x=None, agent_o=None, agent_delay=AGENT_DELAY_MS):
        self.board = board
        self.agent_x = agent_x
        self.agent_o = agent_o
        self.agent_delay = agent_delay
        self.last_move = None
        self.move_count = 0

        # ─── Window setup ───
        self.root = tk.Tk()
        self.root.title("BetaZero — Ultimate Tic-Tac-Toe")
        self.root.configure(bg=BG)
        self.root.resizable(False, False)

        # ─── Fonts ───
        self.font_title = tkfont.Font(family="Courier", size=11, weight="bold")
        self.font_status = tkfont.Font(family="Helvetica", size=11, weight="bold")
        self.font_move_count = tkfont.Font(family="Courier", size=9)
        self.font_piece = tkfont.Font(family="Courier", size=20, weight="bold")
        self.font_big = tkfont.Font(family="Courier", size=40, weight="bold")

        # ─── Header ───
        self.header_frame = tk.Frame(self.root, bg=BG, height=HEADER_H)
        self.header_frame.pack(fill="x", padx=12, pady=(12, 4))

        self.status_label = tk.Label(
            self.header_frame, text="X to move", font=self.font_status,
            fg=X_COLOR, bg=BG, anchor="w"
        )
        self.status_label.pack(side="left")

        self.move_label = tk.Label(
            self.header_frame, text="move 0", font=self.font_move_count,
            fg=TEXT_MUTED, bg=BG, anchor="e"
        )
        self.move_label.pack(side="right")

        # ─── Canvas ───
        self.canvas = tk.Canvas(
            self.root, width=CANVAS_W, height=CANVAS_H,
            bg=BG, highlightthickness=0
        )
        self.canvas.pack(padx=12, pady=(4, 8))
        self.canvas.bind("<Button-1>", self.on_click)

        # ─── Speed slider (only visible when agents are playing) ───
        if agent_x or agent_o:
            self.speed_frame = tk.Frame(self.root, bg=BG)
            self.speed_frame.pack(fill="x", padx=12, pady=(0, 8))
            tk.Label(self.speed_frame, text="delay", font=self.font_move_count,
                     fg=TEXT_MUTED, bg=BG).pack(side="left")
            self.speed_var = tk.IntVar(value=self.agent_delay)
            self.speed_slider = tk.Scale(
                self.speed_frame, from_=50, to=2000, orient="horizontal",
                variable=self.speed_var, bg=BG, fg=TEXT_DIM, troughcolor=SURFACE2,
                highlightthickness=0, length=160, showvalue=True,
                command=lambda v: setattr(self, 'agent_delay', int(v))
            )
            self.speed_slider.pack(side="left", padx=8)
            tk.Label(self.speed_frame, text="ms", font=self.font_move_count,
                     fg=TEXT_MUTED, bg=BG).pack(side="left")

        self.draw_board()
        self.root.after(400, self.engine_move)

    def run(self):
        self.root.mainloop()

    # ─── Coordinate Helpers ───

    def mini_origin(self, board_idx):
        """Top-left corner of a mini-board on the canvas."""
        br, bc = divmod(board_idx, 3)
        x = PAD + bc * (BOARD + PAD)
        y = PAD + br * (BOARD + PAD)
        return x, y

    def cell_rect(self, board_idx, cell_idx):
        """(x1, y1, x2, y2) for a cell."""
        ox, oy = self.mini_origin(board_idx)
        cr, cc = divmod(cell_idx, 3)
        x1 = ox + cc * CELL + 1
        y1 = oy + cr * CELL + 1
        return x1, y1, x1 + CELL - 2, y1 + CELL - 2

    def cell_center(self, board_idx, cell_idx):
        x1, y1, x2, y2 = self.cell_rect(board_idx, cell_idx)
        return (x1 + x2) / 2, (y1 + y2) / 2

    def hit_test(self, mx, my):
        """Return (board_idx, cell_idx) for a mouse click, or None."""
        for bi in range(9):
            for ci in range(9):
                x1, y1, x2, y2 = self.cell_rect(bi, ci)
                if x1 <= mx <= x2 and y1 <= my <= y2:
                    return (bi, ci)
        return None

    # ─── Drawing ───

    def draw_board(self):
        c = self.canvas
        c.delete("all")

        legal_moves = set(tuple(m) for m in self.board.get_legal_moves())
        legal_boards = set(m[0] for m in legal_moves)
        is_terminal = self.board.is_terminal()

        for bi in range(9):
            ox, oy = self.mini_origin(bi)
            macro_val = self.board.macro[bi]

            # ── Mini-board background ──
            bg = SURFACE
            if macro_val == 1:   bg = WON_X_BG
            elif macro_val == -1: bg = WON_O_BG
            elif macro_val == 4:  bg = DRAW_BG
            elif bi in legal_boards and not is_terminal:
                bg = SURFACE2

            c.create_rectangle(ox, oy, ox + BOARD, oy + BOARD,
                               fill=bg, outline=GRID_MAJOR, width=2)

            # ── Active board glow ──
            if bi in legal_boards and not is_terminal and macro_val == 0:
                c.create_rectangle(ox - 1, oy - 1, ox + BOARD + 1, oy + BOARD + 1,
                                   outline=LEGAL_BORDER, width=2)

            # ── Grid lines inside mini-board ──
            for i in range(1, 3):
                c.create_line(ox + i * CELL, oy + 2, ox + i * CELL, oy + BOARD - 2,
                              fill=GRID, width=1)
                c.create_line(ox + 2, oy + i * CELL, ox + BOARD - 2, oy + i * CELL,
                              fill=GRID, width=1)

            for ci in range(9):
                x1, y1, x2, y2 = self.cell_rect(bi, ci)
                cx, cy = self.cell_center(bi, ci)
                val = self.board.boards[bi][ci]

                # ── Legal move highlighting ──
                if (bi, ci) in legal_moves and not is_terminal:
                    c.create_rectangle(x1 + 2, y1 + 2, x2 - 2, y2 - 2,
                                       fill=LEGAL_FILL, outline=LEGAL_BORDER, width=1)

                # ── Last move indicator ──
                if self.last_move and self.last_move == (bi, ci):
                    c.create_rectangle(x1 + 1, y1 + 1, x2 - 1, y2 - 1,
                                       outline=LAST_MOVE, width=2)

                # ── Piece ──
                if val == 1:
                    color = X_COLOR if macro_val == 0 else TEXT_MUTED
                    c.create_text(cx, cy, text="X", font=self.font_piece, fill=color)
                elif val == -1:
                    color = O_COLOR if macro_val == 0 else TEXT_MUTED
                    c.create_text(cx, cy, text="O", font=self.font_piece, fill=color)

            # ── Big overlay symbol for won/drawn boards ──
            if macro_val != 0:
                mcx = ox + BOARD / 2
                mcy = oy + BOARD / 2
                if macro_val == 1:
                    c.create_text(mcx, mcy, text="X", font=self.font_big,
                                  fill=X_COLOR, stipple="gray50")
                elif macro_val == -1:
                    c.create_text(mcx, mcy, text="O", font=self.font_big,
                                  fill=O_COLOR, stipple="gray50")
                elif macro_val == 4:
                    c.create_text(mcx, mcy, text="—", font=self.font_big,
                                  fill=TEXT_MUTED, stipple="gray50")

        # ── Update status ──
        self.update_status()

    def update_status(self):
        result = self.board.check_macro()
        if result == 1:
            self.status_label.config(text="X wins!", fg=X_COLOR)
        elif result == -1:
            self.status_label.config(text="O wins!", fg=O_COLOR)
        elif result == 4:
            self.status_label.config(text="Draw", fg=TEXT_MUTED)
        elif self.board.player == 1:
            self.status_label.config(text="X to move", fg=X_COLOR)
        else:
            self.status_label.config(text="O to move", fg=O_COLOR)

        self.move_label.config(text=f"move {self.move_count}")

    # ─── Interaction ───

    def on_click(self, event):
        if self.board.is_terminal():
            return

        # Block clicks during agent's turn
        if (self.board.player == 1 and self.agent_x) or \
           (self.board.player == -1 and self.agent_o):
            return

        hit = self.hit_test(event.x, event.y)
        if hit is None:
            return

        move = hit
        if move not in self.board.get_legal_moves():
            return

        self.board.apply_move(move)
        self.last_move = move
        self.move_count += 1
        self.draw_board()
        self.root.after(100, self.engine_move)

    def engine_move(self):
        if self.board.is_terminal():
            return

        agent = self.agent_x if self.board.player == 1 else self.agent_o
        if agent is None:
            return

        try:
            move = agent.select_move(self.board)
            if move and move in self.board.get_legal_moves():
                self.board.apply_move(move)
                self.last_move = move
                self.move_count += 1
                self.draw_board()
        except Exception as e:
            print(f"Agent error: {e}")

        if not self.board.is_terminal():
            # Use configurable delay so you can watch agent-vs-agent games
            next_agent = self.agent_x if self.board.player == 1 else self.agent_o
            if next_agent is not None:
                self.root.after(self.agent_delay, self.engine_move)
