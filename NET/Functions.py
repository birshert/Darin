from Field import Field


def possible_moves(field):
    moves = []
    for i in range(field.get_size()):
        for j in range(field.get_size()):
            if field.get_node(i, j).is_empty():
                moves.append({i, j})
    return moves
