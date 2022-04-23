import chess
import numpy as np
from halfkp import flipPers, piece_to_ordinal, flip_piece_pers

def generate_labels(move : chess.Move, board : chess.Board):
    label = np.zeros((2, 384), bool)
    p = piece_to_ordinal(board.piece_at(move.from_square))
    sq = flipPers[move.to_square]

    if board.turn == chess.BLACK:
        p -= 6
        sq = flipPers[sq]

    label[0][(p*64) + (sq)] = True
    for lm in list(board.legal_moves):
        if board.is_capture(lm): continue

        p = piece_to_ordinal(board.piece_at(lm.from_square))
        sq = flipPers[lm.to_square]

        if board.turn == chess.BLACK:
            p -= 6
            sq = flipPers[sq]

        label[1][(p*64) + (sq)] = True

    return label
