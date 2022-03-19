import chess
import numpy as np
from halfkp import flipPers, piece_to_ordinal

def generate_labels(move : chess.Move, board : chess.Board):
    label = np.zeros((384,), bool)
    p = piece_to_ordinal(board.piece_at(move.from_square))
    sq = flipPers[move.to_square]

    label[(p*64) + (sq)] = True
    return label
