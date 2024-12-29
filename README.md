## Symbolic AI CW2

Contributors - Jebastin Nadar, Amit Dhavale

### Running the code
1. To play a game with move recommendations from Minimax algorithm, run
```
python game.py -m 4 -n 3 -k 3
```
2. Use `--prune` to run Minimax algorithm with alpha-beta pruning
```
python game.py -m 4 -n 3 -k 3 --prune
```

3. Use `--benchmark` to get runtime info. The game will be played automatically.
```
python game.py -m 4 -n 3 -k 3 --benchmark
python game.py -m 4 -n 3 -k 3 --prune --benchmark
```