import numpy as np
import chess
import halfkp
import chess.pgn
import labels as Labels

infile = open("data.pgn", 'r')
labels = []

gamesProcessed = 0
featureCount = 0

while gamesProcessed < 20000:
    game = chess.pgn.read_game(infile)
    moves = game.mainline_moves()
    board = chess.Board()
    prevmove = None

    for mid, move in enumerate(moves):
        feature = np.packbits(halfkp.get_halfkp_indeicies(board))
        label = Labels.generate_labels(move, board)
        np.savez_compressed("features/{}".format(featureCount), feature)
        labels.append(label)
        featureCount+=1

        board.push(move)

    gamesProcessed+=1
    print(gamesProcessed)

np.save("labels", np.array(labels))
