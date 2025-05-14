--------------------------------
test1:
--------------------------------
PREDEFINED_OBSTACLES = [
    # Example: (x, y, radius)
    (20, 50, 10),
    (50, 80, 8),
    (70, 40, 12),
    (10, 95, 10),
    (75, 105, 8),
    (-20, 100, 10),
    (-25, 40, 10),
]

LLM Enabled (Physical Obstacles):
- Time: 89.96 seconds
- Iterations: 397 steps
- 
LLM Disabled (Physical Obstacles):
- Time: 119.45 seconds
- Iterations: 489 steps

--------------------------------
test2:
--------------------------------
PREDEFINED_OBSTACLES = [
    # Example: (x, y, radius)
    (20, 50, 10),
    (50, 80, 8),
    (70, 40, 12),
    (10, 95, 10),
    (75, 105, 6),
    (-5, 70, 8),
    (75, 85, 6),
    (-20, 100, 10),
    (-25, 40, 10),
]

LLM Enabled (Physical Obstacles):
- Time: 103.79 seconds
- Iterations: 438 steps

LLM Disabled (Physical Obstacles):
- Time: 133.57 seconds
- Iterations: 489 steps

--------------------------------
test3:
--------------------------------
PREDEFINED_OBSTACLES = [
    # Example: (x, y, radius)
    (-10, 70, 25),
]

- Time: 105.29 seconds
- Iterations: 408 steps