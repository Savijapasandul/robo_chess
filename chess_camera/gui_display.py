import flet as ft

PIECE_SYMBOLS = {
    "white-king": "♔", "white-queen": "♕", "white-rook": "♖",
    "white-bishop": "♗", "white-knight": "♘", "white-pawn": "♙",
    "black-king": "♚", "black-queen": "♛", "black-rook": "♜",
    "black-bishop": "♝", "black-knight": "♞", "black-pawn": "♟"
}

def main(page: ft.Page):
    board_refs = [[ft.Ref[ft.Container]() for _ in range(8)] for _ in range(8)]

    board = ft.Column([
        ft.Row([
            ft.Container(
                ref=board_refs[row][col],
                width=80, height=80,
                bgcolor="#EEE" if (row+col)%2==0 else "#444",
                alignment=ft.alignment.center,
                content=ft.Text("", size=28)
            ) for col in range(8)
        ]) for row in range(8)
    ])

    page.add(board)

    def update_board(pieces):
        for row in range(8):
            for col in range(8):
                board_refs[row][col].current.content = ft.Text("")

        for piece, square in pieces:
            col = ord(square[0]) - ord('a')
            row = 8 - int(square[1])
            symbol = PIECE_SYMBOLS.get(piece, "?")
            board_refs[row][col].current.content = ft.Text(symbol, size=28)

        page.update()

    # Subscribe to updates
    page.pubsub.subscribe(update_board)
