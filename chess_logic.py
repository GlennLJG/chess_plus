"""
Chess Logic - Fonctions Python pour le viewer d'échecs
Chargé et exécuté via Pyodide dans le navigateur
"""

import chess
import chess.pgn
import io
import json
from datetime import datetime


# Symboles Unicode des pièces
PIECE_SYMBOLS = {
    'P': '♙', 'N': '♘', 'B': '♗', 'R': '♖', 'Q': '♕', 'K': '♔',
    'p': '♟', 'n': '♞', 'b': '♝', 'r': '♜', 'q': '♛', 'k': '♚'
}

# Valeur des pièces pour le calcul du matériel
PIECE_VALUES = {'P': 1, 'N': 3, 'B': 3, 'R': 5, 'Q': 9, 'p': 1, 'n': 3, 'b': 3, 'r': 5, 'q': 9}

# Nombre initial de chaque pièce
INITIAL_PIECES = {'P': 8, 'N': 2, 'B': 2, 'R': 2, 'Q': 1, 'p': 8, 'n': 2, 'b': 2, 'r': 2, 'q': 1}


# =============================================================================
# FONCTIONS DE LOGIQUE D'ÉCHECS
# =============================================================================

def get_legal_moves(fen: str, from_square: str) -> tuple[list[str], list[str]]:
    """
    Retourne les coups légaux depuis une case donnée.
    
    Returns:
        Tuple (cases_normales, cases_captures)
    """
    board = chess.Board(fen)
    from_sq = chess.parse_square(from_square)
    
    targets = []
    captures = []
    
    for move in board.legal_moves:
        if move.from_square == from_sq:
            to_sq = chess.square_name(move.to_square)
            if board.piece_at(move.to_square):
                captures.append(to_sq)
            else:
                targets.append(to_sq)
    
    return (targets, captures)


def has_piece(fen: str, square: str) -> bool:
    """Vérifie si une case contient une pièce."""
    board = chess.Board(fen)
    piece = board.piece_at(chess.parse_square(square))
    return piece is not None


def try_move(fen: str, from_square: str, to_square: str) -> str | None:
    """
    Tente de jouer un coup et retourne le nouveau FEN si légal.
    """
    board = chess.Board(fen)
    move_uci = from_square + to_square
    
    try:
        move = chess.Move.from_uci(move_uci)
        if move not in board.legal_moves:
            move = chess.Move.from_uci(move_uci + "q")
        
        if move in board.legal_moves:
            board.push(move)
            return board.fen()
    except:
        pass
    
    return None


def get_position_at_move(moves_list: list[str], move_index: int) -> str:
    """Reconstruit la position après N coups."""
    board = chess.Board()
    for i in range(move_index):
        board.push_san(moves_list[i])
    return board.fen()


def get_initial_fen() -> str:
    """Retourne le FEN de la position initiale."""
    return chess.Board().fen()


def get_board_pieces(fen: str) -> list[dict]:
    """
    Retourne la liste des pièces sur le plateau avec leurs positions.
    """
    board = chess.Board(fen)
    pieces = []
    
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            piece_char = piece.symbol()
            pieces.append({
                'square': chess.square_name(square),
                'piece': piece_char,
                'symbol': PIECE_SYMBOLS.get(piece_char, ''),
                'is_white': piece.color == chess.WHITE
            })
    
    return pieces


def get_captured_pieces(fen: str) -> dict:
    """
    Calcule les pièces capturées et le différentiel de matériel.
    """
    board = chess.Board(fen)
    
    current = {}
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            char = piece.symbol()
            current[char] = current.get(char, 0) + 1
    
    white_captured = ''
    black_captured = ''
    white_material = 0
    black_material = 0
    
    for piece in 'QRBNPqrbnp':
        missing = INITIAL_PIECES.get(piece, 0) - current.get(piece, 0)
        for _ in range(missing):
            if piece.isupper():
                white_captured += PIECE_SYMBOLS[piece]
                black_material += PIECE_VALUES[piece]
            else:
                black_captured += PIECE_SYMBOLS[piece]
                white_material += PIECE_VALUES[piece]
    
    return {
        'white_captured': white_captured,
        'black_captured': black_captured,
        'white_material': white_material,
        'black_material': black_material,
        'diff': white_material - black_material
    }


def get_captured_display(fen: str, flipped: bool) -> dict:
    """
    Retourne les données d'affichage des pièces capturées selon l'orientation.
    
    Args:
        fen: Position en notation FEN
        flipped: True si le plateau est retourné
    
    Returns:
        {
            'top_pieces': 'symboles',
            'top_class': 'white-captured' ou 'black-captured',
            'bottom_pieces': 'symboles',
            'bottom_class': 'white-captured' ou 'black-captured',
            'diff': nombre,
            'show_diff_top': bool,
            'show_diff_bottom': bool
        }
    """
    captured = get_captured_pieces(fen)
    
    white_captured = captured['white_captured']
    black_captured = captured['black_captured']
    diff = captured['diff']
    white_advantage = diff > 0
    
    if flipped:
        # Je joue les noirs (en bas)
        top_pieces = black_captured
        top_class = 'black-captured'
        bottom_pieces = white_captured
        bottom_class = 'white-captured'
        show_diff_top = diff != 0 and white_advantage
        show_diff_bottom = diff != 0 and not white_advantage
    else:
        # Je joue les blancs (en bas)
        top_pieces = white_captured
        top_class = 'white-captured'
        bottom_pieces = black_captured
        bottom_class = 'black-captured'
        show_diff_top = diff != 0 and not white_advantage
        show_diff_bottom = diff != 0 and white_advantage
    
    return {
        'top_pieces': top_pieces,
        'top_class': top_class,
        'bottom_pieces': bottom_pieces,
        'bottom_class': bottom_class,
        'diff': abs(diff),
        'show_diff_top': show_diff_top,
        'show_diff_bottom': show_diff_bottom
    }


def get_game_status(fen: str) -> dict:
    """Retourne le statut de la partie."""
    board = chess.Board(fen)
    return {
        'is_check': board.is_check(),
        'is_checkmate': board.is_checkmate(),
        'is_stalemate': board.is_stalemate(),
        'is_game_over': board.is_game_over(),
        'turn': 'white' if board.turn == chess.WHITE else 'black'
    }


# =============================================================================
# FONCTIONS DE PARSING PGN
# =============================================================================

def parse_pgn(pgn_text: str) -> list[str]:
    """
    Parse un PGN et retourne la liste des coups en notation SAN.
    """
    game = chess.pgn.read_game(io.StringIO(pgn_text))
    
    moves_list = []
    if game:
        node = game
        while node.variations:
            move = node.variation(0).move
            moves_list.append(node.board().san(move))
            node = node.variation(0)
    
    return moves_list


def parse_pgn_headers(pgn_text: str) -> dict:
    """
    Extrait les headers d'un PGN.
    
    Returns:
        Dictionnaire avec White, Black, WhiteElo, BlackElo, Result, etc.
    """
    game = chess.pgn.read_game(io.StringIO(pgn_text))
    
    if game:
        return dict(game.headers)
    return {}


def parse_pgn_full(pgn_text: str) -> dict:
    """
    Parse un PGN complet et retourne toutes les infos.
    
    Returns:
        {
            'moves': [...],
            'headers': {...},
            'white': 'username',
            'black': 'username',
            'white_elo': '1500',
            'black_elo': '1500',
            'result': '1-0'
        }
    """
    game = chess.pgn.read_game(io.StringIO(pgn_text))
    
    if not game:
        return {'moves': [], 'headers': {}}
    
    # Extraire les coups
    moves_list = []
    node = game
    while node.variations:
        move = node.variation(0).move
        moves_list.append(node.board().san(move))
        node = node.variation(0)
    
    headers = dict(game.headers)
    
    return {
        'moves': moves_list,
        'headers': headers,
        'white': headers.get('White', '?'),
        'black': headers.get('Black', '?'),
        'white_elo': headers.get('WhiteElo', '?'),
        'black_elo': headers.get('BlackElo', '?'),
        'result': headers.get('Result', '-')
    }


# =============================================================================
# FONCTIONS API CHESS.COM (appelées via fetch JS, traitées en Python)
# =============================================================================

def parse_game_from_api(game_data: dict) -> dict:
    """
    Parse les données d'une partie de l'API chess.com.
    
    Args:
        game_data: Dictionnaire de l'API chess.com
    
    Returns:
        Dictionnaire formaté pour l'affichage
    """
    white = game_data.get('white', {})
    black = game_data.get('black', {})
    
    white_result = white.get('result', '')
    if white_result == 'win':
        result = '1-0'
    elif white_result in ('checkmated', 'resigned', 'timeout', 'abandoned'):
        result = '0-1'
    else:
        result = '½-½'
    
    end_time = game_data.get('end_time', 0)
    dt = datetime.fromtimestamp(end_time)
    
    return {
        'white': white.get('username', '?'),
        'black': black.get('username', '?'),
        'white_elo': white.get('rating', '?'),
        'black_elo': black.get('rating', '?'),
        'result': result,
        'time_class': game_data.get('time_class', ''),
        'date': dt.strftime('%d/%m/%Y'),
        'time': dt.strftime('%H:%M'),
        'pgn': game_data.get('pgn', '')
    }


def parse_games_list(games_json: str) -> str:
    """
    Parse la liste des parties de l'API chess.com.
    
    Args:
        games_json: JSON string de l'API
    
    Returns:
        JSON string des parties formatées
    """
    data = json.loads(games_json)
    games = data.get('games', [])
    
    # Prendre les 50 dernières, inversées
    games = games[-50:][::-1]
    
    result = []
    for game in games:
        result.append(parse_game_from_api(game))
    
    return json.dumps(result)


def parse_games_from_object(data: dict) -> list[dict]:
    """
    Parse la liste des parties depuis un objet Python (converti depuis JS).
    
    Args:
        data: Dictionnaire avec la clé 'games'
    
    Returns:
        Liste des parties formatées
    """
    games = data.get('games', [])
    
    # Prendre les 50 dernières, inversées
    games = games[-50:][::-1]
    
    result = []
    for game in games:
        result.append(parse_game_from_api(game))
    
    return result


def should_flip_board(username: str, black_player: str) -> bool:
    """
    Détermine si le plateau doit être retourné.
    
    Returns:
        True si l'utilisateur joue les noirs
    """
    return username.lower() == black_player.lower()


def format_player_display(name: str, elo: str) -> str:
    """Formate l'affichage d'un joueur."""
    return f"{name} ({elo})"


# =============================================================================
# FONCTIONS UTILITAIRES
# =============================================================================

def get_square_from_visual(col: int, row: int, flipped: bool) -> str:
    """
    Convertit les coordonnées visuelles en notation échiquéenne.
    
    Args:
        col: Colonne visuelle (0-7, gauche à droite)
        row: Ligne visuelle (0-7, haut en bas)
        flipped: True si le plateau est retourné
    
    Returns:
        Notation de la case (ex: "e4")
    """
    if flipped:
        file_idx = 7 - col
        rank_idx = row
    else:
        file_idx = col
        rank_idx = 7 - row
    
    file = 'abcdefgh'[file_idx]
    rank = rank_idx + 1
    return f"{file}{rank}"


def get_square_center(square: str, flipped: bool) -> dict:
    """
    Retourne le centre d'une case en pixels (pour les flèches).
    
    Args:
        square: Notation de la case (ex: "e4")
        flipped: True si le plateau est retourné
    
    Returns:
        {'x': pixels, 'y': pixels}
    """
    file = square[0]
    rank = int(square[1])
    file_idx = 'abcdefgh'.index(file)
    rank_idx = rank - 1
    
    if flipped:
        col = 7 - file_idx
        row = rank_idx
    else:
        col = file_idx
        row = 7 - rank_idx
    
    return {
        'x': col * 60 + 30,
        'y': row * 60 + 30
    }


def get_board_squares(flipped: bool) -> list[dict]:
    """
    Génère la liste des 64 cases de l'échiquier pour l'affichage.
    
    Args:
        flipped: True si le plateau est retourné
    
    Returns:
        Liste de dictionnaires avec les infos de chaque case
    """
    squares = []
    for row in range(8):
        for col in range(8):
            display_row = row if flipped else 7 - row
            display_col = 7 - col if flipped else col
            file = 'abcdefgh'[display_col]
            rank = display_row + 1
            
            squares.append({
                'square': f"{file}{rank}",
                'is_light': (row + col) % 2 == 0
            })
    
    return squares


def convert_uci_to_san(fen: str, uci_moves: list[str]) -> str:
    """
    Convertit une liste de coups UCI en notation SAN.
    
    Args:
        fen: Position de départ en FEN
        uci_moves: Liste des coups en notation UCI (ex: ["e2e4", "e7e5"])
    
    Returns:
        Chaîne des coups en notation SAN (ex: "e4 e5 Nf3")
    """
    board = chess.Board(fen)
    san_moves = []
    
    for uci in uci_moves:
        try:
            move = chess.Move.from_uci(uci)
            if move in board.legal_moves:
                san_moves.append(board.san(move))
                board.push(move)
            else:
                break
        except:
            break
    
    return ' '.join(san_moves)


def get_last_move(moves_list: list[str], move_index: int) -> dict | None:
    """
    Retourne les cases de départ et d'arrivée du dernier coup joué.
    
    Args:
        moves_list: Liste des coups en notation SAN
        move_index: Index du coup actuel (1-based pour le dernier coup joué)
    
    Returns:
        {'from': 'e2', 'to': 'e4'} ou None si aucun coup
    """
    if move_index <= 0 or not moves_list:
        return None
    
    board = chess.Board()
    last_move = None
    
    for i in range(min(move_index, len(moves_list))):
        move = board.parse_san(moves_list[i])
        last_move = move
        board.push(move)
    
    if last_move:
        return {
            'from': chess.square_name(last_move.from_square),
            'to': chess.square_name(last_move.to_square)
        }
    
    return None
