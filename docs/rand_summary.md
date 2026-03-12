# Random Composer Failure Modes

### 1. **Constant Literals Everywhere**

The random composer samples literals **uniformly** throughout the program structure. It is also constrained by depth so literals tend to be at the leaves when it cannot use any other strategy. This leads to programs filled with hard-coded constants:

```scheme
(λ (x) (range 92 (fold (λ (y z) z) (second []) (droplast 64 x)) ...))
                ^                             ^             ^^
              Constant                    Constant      Constant
```

**Result**: Even if the program uses input `x`, the output is dominated by these constants, producing the same or very similar outputs across different inputs.

### 2. **Fragile Programs with Hardcoded Indices**

Random programs frequently use:
- `nth 87 x` - accesses element 87 (fails on most realistic inputs)
- `replace 46 ...` - replaces index 46
- `droplast 64 x` - drops 64 elements
- `take 52 x` - takes 52 elements

```scheme
Program: (λ (x) (swap (nth 87 x) (last []) (takelast 48 [])))

Outputs:
  []           -> ERROR: nth: index 87 out of bounds
  [1,2,3]      -> ERROR: nth: index 87 out of bounds
  [10,20,30]   -> ERROR: nth: index 87 out of bounds
  [5,4,3,2,1]  -> ERROR: nth: index 87 out of bounds
```

**Result**: Most inputs cause crashes, leaving few successful outputs. With failures excluded from variability calculation, there's often only 0-2 unique outputs.

### 3. **Input-Ignoring Branches**

Random if-statements often have constant conditions that ignore the input:

```scheme
(if false
    (λ (x) <complex expression using x>)
    (λ (y) (filteri (λ (z a) false) [])))
          ^^^^^^^^^^^^^^^^^^^^^^^^^^
          Always returns []
```

**Result**: The program always takes one branch and returns a constant.

### 4. **Degenerate Compositions**

Random sampling creates semantically meaningless compositions:

```scheme
(fold (λ (y z) z) <init> <list>)
      ^^^^^^^^^^^
      Always returns second argument, ignores accumulator
      → Result is always the last element

(fold (λ (y z) 7) <init> x)
      ^^^^^^^^^^^
      Always returns 7
      → Result is always 7 regardless of input
```

**Result**: Complex-looking programs that collapse to constants.
