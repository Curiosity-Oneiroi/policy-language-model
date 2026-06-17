// A tiny, dependency-free chess board: renders a FEN as an 8x8 grid of unicode glyphs,
// highlighting the last move's from/to squares.
const GLYPH = {
  K: "♔", Q: "♕", R: "♖", B: "♗", N: "♘", P: "♙",
  k: "♚", q: "♛", r: "♜", b: "♝", n: "♞", p: "♟",
};

function parseFen(placement) {
  return placement.split("/").map((row) => {
    const r = [];
    for (const ch of row) {
      if (/\d/.test(ch)) for (let i = 0; i < +ch; i++) r.push("");
      else r.push(ch);
    }
    return r;
  });
}

export default function Board({ fen, lastUci }) {
  const placement = (fen || "").split(" ")[0] || "8/8/8/8/8/8/8/8";
  const board = parseFen(placement);
  const from = lastUci ? lastUci.slice(0, 2) : null;
  const to = lastUci ? lastUci.slice(2, 4) : null;
  const cells = [];
  for (let r = 0; r < 8; r++) {
    for (let f = 0; f < 8; f++) {
      const alg = "abcdefgh"[f] + (8 - r);
      const light = (r + f) % 2 === 0;
      const ch = (board[r] || [])[f] || "";
      cells.push(
        <div key={alg} className={`sq ${light ? "light" : "dark"} ${(alg === from || alg === to) ? "last" : ""}`}>
          {GLYPH[ch] || ""}
        </div>
      );
    }
  }
  return <div className="board">{cells}</div>;
}
