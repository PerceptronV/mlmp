# Composer Comparison Summary

## Key Metrics

| Composer | Type Check | Compile | Uses Input | Avg Variability | Avg Size | Avg Depth |
|----------|------------|---------|------------|-----------------|----------|----------|
| random | 100.0% | 100.0% | 93.3% | 0.004 | 114.0 | 7.0 |
| empirical | 66.7% | 100.0% | 93.3% | 0.460 | 13.1 | 5.4 |
| template | 100.0% | 100.0% | 100.0% | 0.803 | 9.4 | 4.5 |
| mcts | 100.0% | 100.0% | 96.7% | 0.069 | 53.2 | 6.6 |
| rule | 100.0% | 100.0% | 97.2% | 0.627 | 12.1 | 5.2 |

## Top Function Usage


### Random

| Function | Count |
|----------|-------|
| `third` | 21 |
| `first` | 20 |
| `+` | 19 |
| `min` | 19 |
| `take` | 17 |
| `length` | 17 |
| `%` | 17 |
| `max` | 17 |
| `product` | 16 |
| `mapi` | 15 |

### Empirical

| Function | Count |
|----------|-------|
| `is_in` | 5 |
| `unique` | 4 |
| `filteri` | 4 |
| `reverse` | 4 |
| `filter` | 4 |
| `fold` | 4 |
| `not` | 4 |
| `is_even` | 4 |
| `cons` | 3 |
| `sort` | 3 |

### Template

| Function | Count |
|----------|-------|
| `cons` | 6 |
| `min` | 6 |
| `singleton` | 6 |
| `first` | 5 |
| `map` | 4 |
| `repeat` | 4 |
| `max` | 4 |
| `%` | 3 |
| `fold` | 3 |
| `append` | 3 |

### Mcts

| Function | Count |
|----------|-------|
| `nth` | 15 |
| `max` | 15 |
| `concat` | 12 |
| `length` | 12 |
| `last` | 12 |
| `-` | 12 |
| `%` | 12 |
| `drop` | 12 |
| `+` | 12 |
| `third` | 11 |

### Rule

| Function | Count |
|----------|-------|
| `singleton` | 60 |
| `first` | 51 |
| `drop` | 49 |
| `map` | 38 |
| `cons` | 35 |
| `==` | 28 |
| `length` | 25 |
| `last` | 22 |
| `append` | 22 |
| `reverse` | 22 |

## Sample Programs


### Random

1. `(if false (λ (x) (range (+ (min (cut_vals 4 [])) (* (second []) (- 69 57))) (nth (sum []) (mapi (λ (y z) y) (repeat 43 48))) (+ (fold (λ (a b) b) (length x) (unique [])) (nth (min x) (splice x 5 x))))) (if (is_in (filter (if (== 46 45) (λ (c) true) (λ (d) true)) (cons (% 81 28) (take 99 []))) (sum (mapi (λ (e f) 72) (reverse [])))) (if (== (foldi (λ (g h i) g) (* 33 54) (reverse [])) (* (sum []) (+ 14 20))) (if (== (if false 70 1) (nth 68 [])) (if (== 20 92) (λ (j) []) (λ (k) [])) (if (is_odd 81) (λ (l) []) (λ (m) []))) (if (is_in (flatten []) (second [])) (λ (n) (unique n)) (λ (o) (cons 10 [])))) (λ (p) (singleton (product (slice 60 21 []))))))`
2. `(if (not (== (third (cut_val (first []) (reverse []))) (== (if true (is_odd 90) (first [])) (fold (λ (x y) true) (== true false) (append [] true))))) (λ (z) (if (not (is_even (max z))) (find (λ (a) (or true true)) (take (+ 59 9) (cut_idx 70 []))) (range (length []) (if (third []) (/ 51 21) (* 33 58)) (* (max z) (min z))))) (λ (b) b))`
3. `(λ (x) (if (last (singleton (and (if true true true) (is_odd 30)))) (range (count (if (is_odd 10) (λ (y) y) (λ (z) z)) (concat (takelast 50 []) (take 96 []))) 79 (count (if (if false false true) (λ (a) false) (λ (b) b)) (cut_slice 70 (- 17 13) (slice 36 91 [])))) (mapi (if (is_even (- 6 54)) (if (if true false false) (λ (c d) 1) (λ (e f) f)) (if (< 47 18) (λ (g h) 46) (λ (i j) j))) (filter (if (> 13 71) (λ (k) true) (λ (l) true)) (sort (λ (m) 3) (cons false []))))))`

### Empirical

1. `(λ (x) [])`
2. `(λ (x) (cons 2 (cons 1 (cons 2 x))))`
3. `(λ (x) (takelast (last x) (sort (λ (y) y) (cut_idx 1 (drop 10 x)))))`

### Template

1. `(λ (x) (cons 94 x))`
2. `(λ (x) (filteri (λ (b c) (< (% c 2) 8)) (mapi (λ (y z) (* y 9)) x)))`
3. `(λ (x) (fold (λ (y z) (cons z y)) [] x))`

### Mcts

1. `(λ (x) (cons (product (range (- (- 8 4) (last x)) (* (product []) (nth 10 x)) (% (third x) (length [])))) (swap (min (concat (reverse x) (singleton 6))) (length x) (cut_slice (% (first []) (% 2 4)) (* (product x) (nth 6 [])) (find (λ (y) true) (range 2 3 8))))))`
2. `(λ (x) (singleton (sum (singleton (max (range 1 6 6))))))`
3. `(λ (x) (range (min []) (third (cut_slice (count (λ (y) false) (drop 3 [])) (min (concat x x)) (cut_slice (last x) (second x) (swap 4 5 x)))) (first (append (cons (last x) (swap 7 1 x)) (max (append x 9))))))`

### Rule

1. `(λ (x) (singleton (third x)))`
2. `(λ (x) (if (> 3 (length x)) [] (singleton (third x))))`
3. `(λ (x) (singleton (nth 7 x)))`
