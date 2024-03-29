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

def getindex(p, sq):
    p = piece_to_ordinal(p)

    if p <= 5:
        sq = flipPers[sq]

    i1 = (64*p) + (sq)

    return i1

def ksqs(pmap):
    wk, bk = 0,0
    for p in pmap:
        if str(pmap[p]) == 'K': wk = p
        if str(pmap[p]) == 'k': bk = p
    
    return wk, bk

def get_halfkp_indeicies(board : chess.Board):
    features = np.zeros(32, dtype=int)
    featureCount = 0
    pmap = board.piece_map()

    for p in pmap:
        i1 = getindex(pmap[p], p)
        features[featureCount] = i1
        featureCount+=1
    
    # return features
    return (features)