import random

# Based on code from http://bernd.bplaced.net/fengenerator/fengenerator.html
class FenGen():
    def make_fen(self) -> str:
        def get_empty_field() -> tuple[int, int]:
            while True:
                x = random.randint(0, 7)
                y = random.randint(0, 7)
                if chb[y][x] == '':
                    return x, y

        def allowed_pos(x:int, y:int, fig:str) -> bool:
            op_king = 'k' if fig.isupper() else 'K'
            
            moves = {
                'K': [(0,-1), (1,-1), (1,0), (1,1), (0,1), (-1,1), (-1,0), (-1,-1)],
                'Q': [(0,-1), (1,-1), (1,0), (1,1), (0,1), (-1,1), (-1,0), (-1,-1)],
                'R': [(0,-1), (1,0), (0,1), (-1,0)],
                'B': [(1,-1), (1,1), (-1,1), (-1,-1)],
                'N': [(1,-2), (2,-1), (2,1), (1,2), (-1,2), (-2,1), (-2,-1), (-1,-2)],
                'P': [(-1,-1), (1,-1)],
            }
            moves['q'], moves['r'], moves['b'], moves['n'], moves['p'] = moves['Q'], moves['R'], moves['B'], moves['N'], [(-1,1), (1,1)]

            for dx, dy in moves[fig.upper()]:
                nx, ny = x + dx, y + dy
                while 0 <= nx < 8 and 0 <= ny < 8:
                    if chb[ny][nx] == op_king:
                        return False
                    if chb[ny][nx] != '':
                        break
                    if fig.upper() in 'KNP':  # These pieces only move one step
                        break
                    nx, ny = nx + dx, ny + dy

            if fig.upper() == 'P' and (y < 1 or y > 6):
                return False

            return True

        chb = [[''] * 8 for _ in range(8)]
        pieces = 'QRBNP' + 'qrbnp'
        piece_counts = [random.randint(0, 1) for _ in range(10)]  # Randomly choose 0 or 1 for each piece type
        
        # Ensure total pieces (including kings) doesn't exceed 6
        while sum(piece_counts) + 2 > 6:
            piece_counts[random.randint(0, 9)] = max(0, piece_counts[random.randint(0, 9)] - 1)

        # Place black king
        bk_x, bk_y = get_empty_field()
        chb[bk_y][bk_x] = 'k'

        # Place white king
        while True:
            wk_x, wk_y = get_empty_field()
            if allowed_pos(wk_x, wk_y, 'K'):
                chb[wk_y][wk_x] = 'K'
                break

        # Place other pieces
        for piece, count in zip(pieces, piece_counts):
            for _ in range(count):
                while True:
                    x, y = get_empty_field()
                    if allowed_pos(x, y, piece):
                        chb[y][x] = piece
                        break

        # Generate FEN string
        fen = ''
        for row in chb:
            empty = 0
            for cell in row:
                if cell == '':
                    empty += 1
                else:
                    if empty > 0:
                        fen += str(empty)
                        empty = 0
                    fen += cell
            if empty > 0:
                fen += str(empty)
            fen += '/'
        fen = fen[:-1] + ' w - - 0 1'  # Remove last '/' and add the rest of FEN string

        return fen

if __name__ == '__main__':
    fenGen = FenGen()
    for _ in range(5):
        print(fenGen.makefen())