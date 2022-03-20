import chess
import numpy as np

flipPers = [56, 57, 58, 59, 60, 61, 62, 63,
  48, 49, 50, 51, 52, 53, 54, 55,
  40, 41, 42, 43, 44, 45, 46, 47,
  32, 33, 34, 35, 36, 37, 38, 39,
  24, 25, 26, 27, 28, 29, 30, 31,
  16, 17, 18, 19, 20, 21, 22, 23,
  8, 9, 10, 11, 12, 13, 14, 15,
  0, 1, 2, 3, 4, 5, 6, 7]

flip_piece_pers = [6, 7, 8, 9, 10, 11, 0, 1, 2, 3, 4, 5]

def piece_to_ordinal(piece):
    return (piece.piece_type - 1) + ((not piece.color) * 6)

def getindex(p, sq, wk, bk, side):
    p = piece_to_ordinal(p)

    if side == chess.WHITE:
        sq = flipPers[sq]
        wk = flipPers[wk]
        bk = flipPers[bk]
    else:
        p = flip_piece_pers[p]
        wk, bk = bk, wk

    return (wk + (768*p) + (64*sq)), (bk + (768*p) + (64*sq) + 49152)

def ksqs(pmap):
    wk, bk = 0,0
    for p in pmap:
        if str(pmap[p]) == 'K': wk = p
        if str(pmap[p]) == 'k': bk = p
    
    return wk, bk

def get_halfkp_indeicies(board : chess.Board):
    features = np.zeros((64*64*12*2), dtype=np.bool)
    pmap = board.piece_map()
    wk, bk = ksqs(pmap)
    for p in pmap:
        i1, i2 = getindex(pmap[p], p, wk, bk, board.turn)
        features[i1] = True
        features[i2] = True
    
    # return features
    return (features)
