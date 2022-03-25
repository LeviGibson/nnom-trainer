import numpy as np
import chess
import halfkp
import chess.pgn
import labels as Labels
import hashlib

infile = open("data.pgn", 'r')

def generate(rows, fname):
    labels = []
    keys = []

    gamesProcessed = 0
    featureCount = 0

    while gamesProcessed < rows:
        game = chess.pgn.read_game(infile)
        moves = game.mainline_moves()
        board = chess.Board()
        prevmove = None

        for mid, move in enumerate(moves):
            if not board.is_capture(move):
                if mid < 7:
                    key = hashlib.md5(bytes(board.fen() + str(move), encoding='ascii')).digest()
                    if key in keys: board.push(move); continue
                    keys.append(key)

                feature = np.packbits(halfkp.get_halfkp_indeicies(board))
                label = Labels.generate_labels(move, board)
                np.savez_compressed(fname + "features/{}".format(featureCount), feature)
                labels.append(label)
                featureCount+=1

            board.push(move)

        gamesProcessed+=1
        print(gamesProcessed)

    np.save(fname + "labels", np.array(labels))

generate(20000, "train_")
generate(500, "val_")
