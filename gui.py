import pygame as pg
import copy
import pprint

# from pygame_widgets.button import Button
from dlgo import goboard
from dlgo.gotypes import Point, Player
from dlgo.utils import print_move
import time

GRID_NUM = 19
WINDOW_SIZE = (1080, 720)
BOARD_LEN = min(WINDOW_SIZE) * 0.8
OFFSET = ((WINDOW_SIZE[0]-BOARD_LEN)/2, (WINDOW_SIZE[1]-BOARD_LEN)/2)
GRID_LEN = BOARD_LEN / (GRID_NUM-1)
STONE_RADIUS = GRID_LEN * 0.5
X_CORDS = [chr(i) for i in range(ord('A'), ord('A') + GRID_NUM)]  # A-S for 19x19

class GameGUI:
    def __init__(self, game, bot=None, bot_players=None):
        pg.init()
        self.game = game
        self.bot = bot
        self.bot_players = bot_players or []
        self.window = pg.display.set_mode(WINDOW_SIZE)
        # self.buttons = self.create_buttons()
        self.board_surface = None        
        self.display_board(game)

    # def create_buttons(self):
    #     return [
    #         Button(
    #         self.window, 50, 100, 100, 50, text='Undo',
    #         fontSize=30, margin=20,
    #         inactiveColour=(147, 153, 152),
    #         pressedColour=(147, 153, 152), radius=20,
    #         onClick=self.game.undo
    #     ),
    #         Button(
    #         self.window, 50, 325, 100, 50, text='Pass',
    #         fontSize=30, margin=20,
    #         inactiveColour=(147, 153, 152),
    #         pressedColour=(147, 153, 152), radius=20,
    #         onClick=self.game.player_pass
    #     ),
    #         Button(
    #         self.window, 50, 550, 100, 50, text='Resign',
    #         fontSize=30, margin=20,
    #         inactiveColour=(147, 153, 152),
    #         pressedColour=(147, 153, 152), radius=20,
    #         onClick=self.game.resign
    #     )
    #     ]

    def display_board(self, state):
        """
        Draw everything:
        - Wooden background
        - Grid lines
        - Coordinates
        - Stones from gameboard
        - Mark the most recent move
        """
        board = pg.Surface(WINDOW_SIZE)
        
        # Background color
        area = (OFFSET[0] - GRID_LEN, OFFSET[1] - GRID_LEN,
                2*GRID_LEN + BOARD_LEN, 2*GRID_LEN + BOARD_LEN)
        board.fill(color=(125, 125, 125))  # dark grey background
        board.fill(color=(242, 194, 111), rect=area)  # "wooden" color for board
        
        # Draw board boundary lines
        pg.draw.rect(board, (0, 0, 0), area, width=2)
        
        font = pg.font.SysFont('chalkduster.ttf', 20)
        
        # Draw coordinate labels (A-S, 1-19)
        # Horizontal (X) labels
        space = 0
        for cord in X_CORDS:
            text = font.render(cord, True, (0,0,0))
            board.blit(text, (OFFSET[0] + space - 6, BOARD_LEN+OFFSET[1] + STONE_RADIUS))
            board.blit(text, (OFFSET[0] + space - 6, OFFSET[1] - 2*STONE_RADIUS))
            space += GRID_LEN
        
        # Vertical (Y) labels
        space = 0
        for i in range(GRID_NUM):
            text = font.render(str(i+1), True, (0,0,0))
            board.blit(text, (OFFSET[0] - 2*STONE_RADIUS,
                              BOARD_LEN+OFFSET[1] - 8 - space))
            board.blit(text, (BOARD_LEN + OFFSET[0] + STONE_RADIUS,
                              BOARD_LEN+OFFSET[1] - 8 - space))
            space += GRID_LEN
        
        # Draw grid lines
        for i in range(GRID_NUM):
            # vertical line
            start_x = OFFSET[0] + i * GRID_LEN
            pg.draw.line(board, (0,0,0),
                         (start_x, OFFSET[1]),
                         (start_x, OFFSET[1] + BOARD_LEN))
            
            # horizontal line
            start_y = OFFSET[1] + i * GRID_LEN
            pg.draw.line(board, (0,0,0),
                         (OFFSET[0], start_y),
                         (OFFSET[0] + BOARD_LEN, start_y))
        
        # Draw star points (3x3, 3x9, 3x15, etc.)
        star_offsets = [3, 9, 15] if GRID_NUM == 19 else []
        for sx in star_offsets:
            for sy in star_offsets:
                px = OFFSET[0] + sx * GRID_LEN
                py = OFFSET[1] + sy * GRID_LEN
                pg.draw.circle(board, (0,0,0), (px, py), 4)
        
        # Draw stones from self.gameboard
        for row in range(1, GRID_NUM + 1):
            for col in range(1, GRID_NUM + 1):
                point = Point(row, col)
                color = state.board.get(point)
                if color == Player.black:
                    px = OFFSET[0] + (col - 1) * GRID_LEN
                    py = OFFSET[1] + (row - 1) * GRID_LEN
                    pg.draw.circle(board, (0,0,0), (px,py), STONE_RADIUS)
                elif color == Player.white:
                    px = OFFSET[0] + (col - 1) * GRID_LEN
                    py = OFFSET[1] + (row - 1) * GRID_LEN
                    pg.draw.circle(board, (255,255,255), (px,py), STONE_RADIUS)

        # Display Captures
        # cap_font = pg.font.SysFont('chalkduster.ttf', 30)
        # b_cap = cap_font.render("Black Captures: %d" % (state.captures['black']), True, (0,0,0))
        # w_cap = cap_font.render("White Captures: %d" % (state.captures['white']), True, (0,0,0))
        # board.blit(b_cap, (0.82*WINDOW_SIZE[0], 0.4*WINDOW_SIZE[1]))
        # board.blit(w_cap, (0.82*WINDOW_SIZE[0], 0.5*WINDOW_SIZE[1]))

        # Mark most recent move 
        if state.last_move and state.last_move.is_play:
            stone = (
                OFFSET[0] + (state.last_move.point.col - 1) * GRID_LEN,  # col index
                OFFSET[1] + (state.last_move.point.row - 1) * GRID_LEN   # row index
            )
            if state.next_player == Player.black:  # Mark on white stone
                pg.draw.circle(board, (0, 0, 0), stone, 0.6 * STONE_RADIUS)
                pg.draw.circle(board, (255, 255, 255), stone, 0.4 * STONE_RADIUS)
            else:  # Mark on black stone
                pg.draw.circle(board, (255, 255, 255), stone, 0.6 * STONE_RADIUS)
                pg.draw.circle(board, (0, 0, 0), stone, 0.4 * STONE_RADIUS)
        
        # Blit to the main window
        self.window.blit(board, (0, 0))
        pg.display.update()
        
        self.board_surface = board

    def run(self):
        """
        Main loop (simplified). In your actual code, you'd probably integrate
        this with event handling for Pygame. 
        """
        running = True
        clock = pg.time.Clock()
        
        while running:
            clock.tick(30)
            events = pg.event.get()
            
            if self.game.next_player in self.bot_players and not self.game.is_over():
                # move = self.bot.select_move(self.game)
                move = self.bot[self.game.next_player].select_move(self.game)
                print_move(self.game.next_player, move)
                self.game = self.game.apply_move(move)
                self.display_board(self.game)
                pg.time.wait(10)  # short pause to show bot move
                continue

            for event in events:
                if event.type == pg.QUIT:
                    running = False
                
                elif event.type == pg.MOUSEBUTTONDOWN:
                    # Translate click position into board row/col
                    mx, my = pg.mouse.get_pos()
                    col = round((mx - OFFSET[0]) / GRID_LEN)
                    row = round((my - OFFSET[1]) / GRID_LEN)
                    click_point = Point(row + 1, col + 1)

                    if self.game.next_player not in self.bot_players:
                        mx, my = pg.mouse.get_pos()
                        col = round((mx - OFFSET[0]) / GRID_LEN)
                        row = round((my - OFFSET[1]) / GRID_LEN)
                        click_point = Point(row + 1, col + 1)

                        if self.game.board.is_on_grid(click_point):
                            move = goboard.Move.play(click_point)
                            if self.game.is_valid_move(move):
                                self.game = self.game.apply_move(move)
                                self.display_board(self.game)
            # Update the button states
            """self.undo_button.listen(events)
            self.undo_button.draw()
            
            self.pass_button.listen(events)
            self.pass_button.draw()
            
            self.resign_button.listen(events)
            self.resign_button.draw()"""
            self.window.blit(self.board_surface, (0, 0))

            mx, my = pg.mouse.get_pos()
            if OFFSET[0] - STONE_RADIUS <= mx <= BOARD_LEN + OFFSET[0] + STONE_RADIUS and OFFSET[1] - STONE_RADIUS <= my <= BOARD_LEN + OFFSET[1] + STONE_RADIUS:
                # Round to snap the cursor stone into place
                col = round((mx - OFFSET[0]) / GRID_LEN)
                row = round((my - OFFSET[1]) / GRID_LEN)
                point = Point(row+1, col+1)
                color = self.game.board.get(point)
                if self.game.board.is_on_grid(point) and color is None:
                    color = (0, 0, 0) if self.game.next_player == Player.black else (255, 255, 255)
                    px = OFFSET[0] + col * GRID_LEN
                    py = OFFSET[1] + row * GRID_LEN
                    pg.draw.circle(self.window, color, (px, py), STONE_RADIUS)

            pg.display.flip()
        
        pg.quit()