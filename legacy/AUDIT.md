# Audit of the v1 results

The v1 README and paper reported 9-0-1 and 8-0-2 records for DataGo against KataGo. Those records do not measure playing strength. The match runner had a protocol bug that gave DataGo extra moves, and its retrieval path never reached the board. This note documents what the code did, with a reproduction, so the old numbers are not mistaken for evidence.

All paths below are under `legacy/v1/datago/`.

## 1. Every deep search placed an extra stone

`run_datago_recursive_match.py` analysed positions by sending KataGo the GTP command `kata-genmove_analyze b` (`src/bot/gtp_controller.py`, `genmove_analyze`). That command analyses the position and also plays the generated move on KataGo's board.

`generate_move` called it once for the standard search. When the uncertainty gate fired, `run_deep_augmented_search` called it again for the deep search, and once more for each recursive child search. White was asked to move only after all of those calls returned. Each deep search therefore put one more black stone on the board within a single black turn.

`legacy/audit/reproduce_double_move.py` runs v1's own controller against KataGo 1.18.2 with the b18c384nbt network:

```
Black 'searches' returned: Q16, Q4, Q5. White then played: D16.
Stones on KataGo's board: 3 black, 1 white.
One black turn placed three stones.
```

The v1 logs fit this exactly. Wins track the number of deep searches:

| v1 experiment | Deep searches (10 games) | Record |
|---|---|---|
| C: real network, threshold 0.37 | 0 | 0-10-0 |
| D: real network, threshold 0.15 | 328 | 8-0-2 |
| B: "synthetic", threshold 0.37 | 1,411 | 9-0-1 |

Experiment D logged 454 moves, so 227 black turns shared 328 extra stones. With no deep searches the same code lost all ten games. That is consistent with one engine playing itself: KataGo rates Black's chances from the empty board at 37% with komi 7.5 under these rules.

## 2. Retrieved moves were never played

On a cache hit, `generate_move` set `move = best_ctx['move']` and updated its own `BoardState`. It never sent that move to KataGo. The stone on the real board was the one `kata-genmove_analyze` had already placed. Retrieval changed the bookkeeping and nothing else.

## 3. Other problems in the same runner

- **One engine played both sides.** DataGo and KataGo were the same process and the same search tree. `kata-set-param maxVisits` set for a deep search stayed in force for White's reply.
- **The reported move could differ from the played move.** `genmove_analyze` returned the first `info move` entry of the analysis stream. KataGo's actual choice is on the final `play` line. In the reproduction above the parser reported Q5 for a search that played a different point.
- **No captures in the position tracker.** `BoardState.play_move` only writes a stone. After the first capture, the symmetry hash described a position that was not on the board.
- **Half of the uncertainty score was always zero.** The value-spread term read `analysis['raw_analysis']`, a key that was never set, so `K` was 0 and the score reduced to `0.5 * E * phase`.
- **The ANN embedding was the first 64 policy entries**, which is the top three rows of the board plus seven points.
- **Games were capped at 100 moves** and scored with `final_score` on an unfinished board. A double pass was recorded as a draw.
- **"Synthetic" experiments used generated network outputs**, as v1's own `SYNTHETIC_VS_REAL_NN_COMPARISON.md` says ("previous synthetic/random data").

## 4. What carries over to v2

The questions were good ones: when is extra search worth paying for, and can stored searches be reused? v2 keeps those questions and rebuilds everything underneath them: a rules-complete board, separate engines per player, exact visit accounting, and matches large enough to carry confidence intervals. The v1 code, paper, slides, and logs are kept unchanged in `legacy/v1/` for the record.
