from logging import CRITICAL
import chess
import tensorflow as tf
import tensorflow.keras as keras
import numpy as np
from labels import generate_labels

from halfkp import get_halfkp_indeicies

model = keras.models.load_model("production/")

def printBestMove(fen):
    board = chess.Board(fen)
    indicies = get_halfkp_indeicies(board)
    x = np.zeros((64*64*12*2,), bool)
    np.add.at(x, indicies, indicies.astype(bool))
    x = (np.array([x, x]), np.ones((2,384,)))
    pred = model.predict(x)

    maxeval = -100000
    maxMove = None

    for move in board.legal_moves:
        if board.is_capture(move): continue
        eval = pred[0][generate_labels(move, board)[0]]
        if eval > maxeval: maxeval = eval; maxMove = move

    print("bestmove", maxMove)

UciBoard = chess.Board()

while True:
    command = input()

    if command == 'quit':
        exit()

    if command == 'uci':
        print("id name Random NNOM wheeee")
        print("id name Levi Gibson")
        print("uciok")

    if command == 'isready':
        print("readyok")

    command = command.split(' ')

    if command[0] == 'go':
        printBestMove(UciBoard.fen())

    if command[0] == 'position':
        if command[1] == 'startpos': UciBoard.set_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
        if command[1] == 'fen': UciBoard.set_fen(command[2])
        if 'moves' in command:
            print(UciBoard)

            fm = False
            for part in command:
                if fm:
                    UciBoard.push_uci(part)
                if part == 'moves':
                    fm = True
