import numpy as np
import chess
import halfkp
import chess.pgn
import labels as Labels
import hashlib
import subprocess
from npy_append_array import NpyAppendArray

infile = open("data.pgn", 'r')
subprocess.call("rm *.npy", shell=True)

def generate(rows, fname):
    features = NpyAppendArray(fname + "features.npy")
    labels = NpyAppendArray(fname + "labels.npy")

    gamesProcessed = 0
    featureCount = 0

    while gamesProcessed < rows:
        game = chess.pgn.read_game(infile)
        moves = game.mainline_moves()
        board = chess.Board()

        for mid, move in enumerate(moves):
            if not board.is_capture(move) and board.turn == chess.WHITE:
                feature = halfkp.get_halfkp_indeicies(board)
                label = Labels.generate_labels(move, board)
                features.append(np.array([feature]))
                labels.append(np.array([label]))
                featureCount+=1

            board.push(move)

        gamesProcessed+=1
        print(gamesProcessed)

generate(100000, "train_")
generate(500, "val_")
