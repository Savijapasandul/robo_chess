import tkinter as tk
from gui import ChessGUI
from logic import ChessLogic
from engine import StockfishEngine
        
def main():
    print("Welcome to RoboChess")
    
    root = tk.Tk()
    root.title("RoboChess")
    logic = ChessLogic()
    engine = StockfishEngine()
    gui = ChessGUI(root, logic, engine)
    root.mainloop()

if __name__ == '__main__':
    main()
