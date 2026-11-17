"""
Algorithmic Problem Taxonomy & Complexity Registry
AI-Based Automated Code Grading and Algorithmic Complexity Prediction Using AST and ANN

Maps competitive programming problem IDs (derived from Project CodeNet / Aizu Online Judge / AtCoder)
to their canonical problem specifications, algorithmic paradigms, input constraints,
and ground-truth asymptotic Time and Space Complexity classes.

Ground Truth Complexity Taxonomy:
- Time Complexity Classes: O(1), O(log N), O(N), O(N log N), O(N^2), O(N^3), O(2^N)
- Space Complexity Classes: O(1), O(log N), O(N), O(N^2)
"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict


@dataclass(frozen=True)
class ProblemSpec:
    problem_id: str
    title: str
    algorithmic_paradigm: str
    time_complexity: str
    space_complexity: str
    description: str
    input_constraints: str
    sample_solutions: List[str]


# Algorithmic Problem Definitions with authentic competitive programming specifications
PROBLEM_REGISTRY: Dict[str, ProblemSpec] = {
    # -------------------------------------------------------------
    # 1. Constant Time: O(1)
    # -------------------------------------------------------------
    "p00001": ProblemSpec(
        problem_id="p00001",
        title="List of Top 3 Hills",
        algorithmic_paradigm="Direct Selection / Constant Evaluation",
        time_complexity="O(1)",
        space_complexity="O(1)",
        description="Given fixed 10 mountain heights, output the top 3 highest in descending order.",
        input_constraints="Exactly 10 integers, each between 0 and 10000.",
        sample_solutions=[
            """# Solution 1: Direct sorting of fixed 10 elements
heights = [int(input()) for _ in range(10)]
heights.sort(reverse=True)
for h in heights[:3]:
    print(h)
""",
            """# Solution 2: Min-heap with fixed bound
import heapq
h = []
for _ in range(10):
    val = int(input())
    if len(h) < 3:
        heapq.heappush(h, val)
    else:
        heapq.heappushpop(h, val)
for val in sorted(h, reverse=True):
    print(val)
""",
            """# Solution 3: Direct 3-variable tracking
first = second = third = -1
for _ in range(10):
    x = int(input())
    if x > first:
        first, second, third = x, first, second
    elif x > second:
        second, third = x, second
    elif x > third:
        third = x
print(first)
print(second)
print(third)
"""
        ]
    ),
    "p00003": ProblemSpec(
        problem_id="p00003",
        title="Is it A Right Triangle?",
        algorithmic_paradigm="Geometry / Pythagorean Arithmetic",
        time_complexity="O(1)",
        space_complexity="O(1)",
        description="Determine whether a triangle with sides a, b, c is a right-angled triangle.",
        input_constraints="1 <= a, b, c <= 1000",
        sample_solutions=[
            """# Solution 1: Sort sides and check Pythagorean theorem
import sys
for line in sys.stdin:
    parts = list(map(int, line.split()))
    if not parts:
        break
    parts.sort()
    if parts[0]**2 + parts[1]**2 == parts[2]**2:
        print("YES")
    else:
        print("NO")
""",
            """# Solution 2: Function-based check
def is_right(a, b, c):
    sides = sorted([a, b, c])
    return sides[0] * sides[0] + sides[1] * sides[1] == sides[2] * sides[2]

n = int(input())
for _ in range(n):
    a, b, c = map(int, input().split())
    print("YES" if is_right(a, b, c) else "NO")
"""
        ]
    ),
    "p00004": ProblemSpec(
        problem_id="p00004",
        title="Simultaneous Equation",
        algorithmic_paradigm="Linear Algebra / Cramer's Rule",
        time_complexity="O(1)",
        space_complexity="O(1)",
        description="Solve 2x2 system of linear equations ax + by = c and dx + ey = f.",
        input_constraints="-1000 <= a,b,c,d,e,f <= 1000",
        sample_solutions=[
            """# Solution 1: Determinant method (Cramer's rule)
import sys
for line in sys.stdin:
    if not line.strip():
        continue
    a, b, c, d, e, f = map(float, line.split())
    det = a * e - b * d
    x = (c * e - b * f) / det
    y = (a * f - c * d) / det
    print(f"{x + 1e-9:.3f} {y + 1e-9:.3f}")
"""
        ]
    ),
    "p00021": ProblemSpec(
        problem_id="p00021",
        title="Parallelism",
        algorithmic_paradigm="Vector Math / 2D Cross Product",
        time_complexity="O(1)",
        space_complexity="O(1)",
        description="Determine whether segment AB is parallel to segment CD.",
        input_constraints="Coordinates are floating point numbers.",
        sample_solutions=[
            """# Solution 1: 2D Cross product of direction vectors
def is_parallel(x1, y1, x2, y2, x3, y3, x4, y4):
    dx1 = x2 - x1
    dy1 = y2 - y1
    dx2 = x4 - x3
    dy2 = y4 - y3
    cross = dx1 * dy2 - dy1 * dx2
    return abs(cross) < 1e-10

n = int(input())
for _ in range(n):
    pts = list(map(float, input().split()))
    print("YES" if is_parallel(*pts) else "NO")
"""
        ]
    ),
    "p00216": ProblemSpec(
        problem_id="p00216",
        title="Cutting Water Off",
        algorithmic_paradigm="Tiered Formula / Piecewise Arithmetic",
        time_complexity="O(1)",
        space_complexity="O(1)",
        description="Calculate water bill given usage volume in cubic meters across progressive tariff brackets.",
        input_constraints="0 <= w <= 100",
        sample_solutions=[
            """# Solution 1: Piecewise tier calculation
def compute_cost(w):
    base = 1150
    extra = 0
    if w > 30:
        extra += (w - 30) * 160
        w = 30
    if w > 20:
        extra += (w - 20) * 140
        w = 20
    if w > 10:
        extra += (w - 10) * 125
    return 4280 - (base + extra)

while True:
    w = int(input())
    if w == -1:
        break
    print(compute_cost(w))
"""
        ]
    ),

    # -------------------------------------------------------------
    # 2. Logarithmic Time: O(log N)
    # -------------------------------------------------------------
    "p00005": ProblemSpec(
        problem_id="p00005",
        title="GCD and LCM",
        algorithmic_paradigm="Number Theory / Euclidean Algorithm",
        time_complexity="O(log N)",
        space_complexity="O(1)",
        description="Compute Greatest Common Divisor and Least Common Multiple of two integers.",
        input_constraints="1 <= a, b <= 2,000,000,000",
        sample_solutions=[
            """# Solution 1: Iterative Euclidean Algorithm
import sys

def gcd(x, y):
    while y != 0:
        x, y = y, x % y
    return x

def lcm(x, y):
    return (x // gcd(x, y)) * y

for line in sys.stdin:
    if not line.strip():
        break
    a, b = map(int, line.split())
    g = gcd(a, b)
    l = lcm(a, b)
    print(f"{g} {l}")
""",
            """# Solution 2: Recursive Euclidean Algorithm
import math

def solve(a, b):
    g = math.gcd(a, b)
    l = (a * b) // g
    return g, l

while True:
    try:
        a, b = map(int, input().split())
        g, l = solve(a, b)
        print(g, l)
    except (EOFError, ValueError):
        break
"""
        ]
    ),
    "p00052": ProblemSpec(
        problem_id="p00052",
        title="Factorial II (Trailing Zeros)",
        algorithmic_paradigm="Legendre's Formula / Logarithmic Division",
        time_complexity="O(log N)",
        space_complexity="O(1)",
        description="Compute number of trailing zeros in N! by counting factors of 5.",
        input_constraints="1 <= n <= 200,000,000",
        sample_solutions=[
            """# Solution 1: Repeated division by powers of 5
while True:
    n = int(input())
    if n == 0:
        break
    count = 0
    power = 5
    while power <= n:
        count += n // power
        power *= 5
    print(count)
""",
            """# Solution 2: In-place division loop
def trailing_zeros(n):
    zeros = 0
    while n >= 5:
        zeros += n // 5
        n //= 5
    return zeros

while True:
    val = int(input())
    if val == 0:
        break
    print(trailing_zeros(val))
"""
        ]
    ),
    "p00044": ProblemSpec(
        problem_id="p00044",
        title="Prime Number II (Binary Search on Primes)",
        algorithmic_paradigm="Binary Search / Sieve Precomputation",
        time_complexity="O(log N)",
        space_complexity="O(1)",
        description="Given integer n, find greatest prime strictly smaller than n and smallest prime strictly larger than n.",
        input_constraints="3 <= n <= 50,000",
        sample_solutions=[
            """# Solution 1: Binary search with bisect
import bisect

def is_prime(x):
    if x < 2: return False
    for d in range(2, int(x**0.5) + 1):
        if x % d == 0:
            return False
    return True

primes = [p for p in range(2, 60000) if is_prime(p)]

while True:
    try:
        n = int(input())
        idx = bisect.bisect_left(primes, n)
        prev_p = primes[idx - 1]
        next_p = primes[idx] if primes[idx] > n else primes[idx + 1]
        print(f"{prev_p} {next_p}")
    except (EOFError, IndexError):
        break
"""
        ]
    ),
    "p00158": ProblemSpec(
        problem_id="p00158",
        title="Collatz Problem",
        algorithmic_paradigm="Iterative Simulation / Logarithmic Sequence",
        time_complexity="O(log N)",
        space_complexity="O(1)",
        description="Compute number of operational steps to reduce integer n to 1 via 3n+1 rules.",
        input_constraints="1 <= n <= 1,000,000",
        sample_solutions=[
            """# Solution 1: Standard Collatz loop
def collatz_steps(n):
    steps = 0
    while n != 1:
        if n % 2 == 0:
            n //= 2
        else:
            n = 3 * n + 1
        steps += 1
    return steps

while True:
    num = int(input())
    if num == 0:
        break
    print(collatz_steps(num))
"""
        ]
    ),
    "p00045": ProblemSpec(
        problem_id="p00045",
        title="Recursive Binary Search",
        algorithmic_paradigm="Divide and Conquer / Logarithmic Recursion Stack",
        time_complexity="O(log N)",
        space_complexity="O(log N)",
        description="Recursively locate target integer in sorted sequence, utilizing O(log N) stack frames.",
        input_constraints="Sorted array up to 100,000 integers.",
        sample_solutions=[
            """# Solution 1: Recursive binary search with O(log N) recursion call stack
def rec_binary_search(arr, low, high, target):
    if low > high:
        return -1
    mid = (low + high) // 2
    if arr[mid] == target:
        return mid
    elif arr[mid] > target:
        return rec_binary_search(arr, low, mid - 1, target)
    else:
        return rec_binary_search(arr, mid + 1, high, target)

import sys
lines = sys.stdin.read().split()
if lines:
    n = int(lines[0])
    arr = list(map(int, lines[1:n+1]))
    target = int(lines[n+1])
    print(rec_binary_search(arr, 0, n - 1, target))
"""
        ]
    ),

    # -------------------------------------------------------------
    # 3. Linear Time: O(N)
    # -------------------------------------------------------------
    "p00006": ProblemSpec(
        problem_id="p00006",
        title="Reverse Sequence",
        algorithmic_paradigm="Single-Pass String Traversal",
        time_complexity="O(N)",
        space_complexity="O(N)",
        description="Reverse a string of length N.",
        input_constraints="String length up to 1000 characters.",
        sample_solutions=[
            """# Solution 1: Slice reversal
s = input().strip()
print(s[::-1])
""",
            """# Solution 2: Explicit two-pointer list swap
chars = list(input().strip())
left, right = 0, len(chars) - 1
while left < right:
    chars[left], chars[right] = chars[right], chars[left]
    left += 1
    right -= 1
print("".join(chars))
"""
        ]
    ),
    "p00007": ProblemSpec(
        problem_id="p00007",
        title="Debt Hell",
        algorithmic_paradigm="Iterative Simulation / Weekly Compounding",
        time_complexity="O(N)",
        space_complexity="O(1)",
        description="Simulate compound interest over n weeks, rounding up to nearest 1,000 yen.",
        input_constraints="1 <= n <= 100",
        sample_solutions=[
            """# Solution 1: Weekly iteration with round-up
def compute_debt(weeks):
    debt = 100000
    for _ in range(weeks):
        interest = int(debt * 0.05)
        debt += interest
        rem = debt % 1000
        if rem != 0:
            debt += (1000 - rem)
    return debt

n = int(input())
print(compute_debt(n))
"""
        ]
    ),
    "p00015": ProblemSpec(
        problem_id="p00015",
        title="National Budget",
        algorithmic_paradigm="High-Precision Integer Addition",
        time_complexity="O(N)",
        space_complexity="O(N)",
        description="Add two arbitrarily large integers up to 80 digits.",
        input_constraints="Number of digits up to 80.",
        sample_solutions=[
            """# Solution 1: Direct big integer addition with overflow check
n = int(input())
for _ in range(n):
    a = int(input())
    b = int(input())
    res = a + b
    if len(str(res)) > 80:
        print("overflow")
    else:
        print(res)
""",
            """# Solution 2: Manual digit-by-digit addition
def add_strings(s1, s2):
    i = len(s1) - 1
    j = len(s2) - 1
    carry = 0
    res = []
    while i >= 0 or j >= 0 or carry:
        d1 = int(s1[i]) if i >= 0 else 0
        d2 = int(s2[j]) if j >= 0 else 0
        total = d1 + d2 + carry
        res.append(str(total % 10))
        carry = total // 10
        i -= 1
        j -= 1
    ans = "".join(reversed(res))
    return "overflow" if len(ans) > 80 else ans
"""
        ]
    ),
    "p00100": ProblemSpec(
        problem_id="p00100",
        title="Initial Sales",
        algorithmic_paradigm="Hash Map Aggregation / Linear Scan",
        time_complexity="O(N)",
        space_complexity="O(N)",
        description="Given N employee sales records, compute total revenue per employee and output those exceeding 1,000,000 yen.",
        input_constraints="1 <= n <= 4000",
        sample_solutions=[
            """# Solution 1: Dict aggregation preserving insertion order
while True:
    n = int(input())
    if n == 0:
        break
    sales = {}
    order = []
    for _ in range(n):
        eid, price, qty = map(int, input().split())
        revenue = price * qty
        if eid not in sales:
            sales[eid] = 0
            order.append(eid)
        sales[eid] += revenue
    
    qualifiers = [eid for eid in order if sales[eid] >= 1000000]
    if not qualifiers:
        print("NA")
    else:
        for q in qualifiers:
            print(q)
"""
        ]
    ),

    # -------------------------------------------------------------
    # 4. Linearithmic Time: O(N log N)
    # -------------------------------------------------------------
    "p00009": ProblemSpec(
        problem_id="p00009",
        title="Prime Number (Sieve of Eratosthenes)",
        algorithmic_paradigm="Number Theory / Sieve Precomputation",
        time_complexity="O(N log N)",
        space_complexity="O(N)",
        description="Count the number of primes up to integer n.",
        input_constraints="1 <= n <= 999,999",
        sample_solutions=[
            """# Solution 1: Sieve of Eratosthenes with prefix counts
import sys

MAX_V = 1000000
is_prime = [True] * MAX_V
is_prime[0] = is_prime[1] = False
for i in range(2, int(MAX_V**0.5) + 1):
    if is_prime[i]:
        for j in range(i * i, MAX_V, i):
            is_prime[j] = False

prime_count = [0] * MAX_V
for i in range(1, MAX_V):
    prime_count[i] = prime_count[i-1] + (1 if is_prime[i] else 0)

for line in sys.stdin:
    if line.strip():
        n = int(line.strip())
        print(prime_count[n])
"""
        ]
    ),
    "p00051": ProblemSpec(
        problem_id="p00051",
        title="Differential",
        algorithmic_paradigm="Sorting / Array Rearrangement",
        time_complexity="O(N log N)",
        space_complexity="O(N)",
        description="Form maximum and minimum integers by sorting 8 digits, then compute their difference.",
        input_constraints="8-digit string.",
        sample_solutions=[
            """# Solution 1: Dual sorting
t = int(input())
for _ in range(t):
    digits = list(input().strip())
    max_val = int("".join(sorted(digits, reverse=True)))
    min_val = int("".join(sorted(digits)))
    print(max_val - min_val)
"""
        ]
    ),
    "p00180": ProblemSpec(
        problem_id="p00180",
        title="Stellar Craft (Minimum Spanning Tree)",
        algorithmic_paradigm="Graph Theory / Kruskal's Algorithm",
        time_complexity="O(N log N)",
        space_complexity="O(N)",
        description="Find weight of Minimum Spanning Tree connecting n stars with m bridges.",
        input_constraints="2 <= n <= 100, 1 <= m <= 1000",
        sample_solutions=[
            """# Solution 1: Kruskal's MST using Disjoint Set Union
class DSU:
    def __init__(self, size):
        self.parent = list(range(size))
    def find(self, i):
        if self.parent[i] == i:
            return i
        self.parent[i] = self.find(self.parent[i])
        return self.parent[i]
    def union(self, i, j):
        root_i = self.find(i)
        root_j = self.find(j)
        if root_i != root_j:
            self.parent[root_i] = root_j
            return True
        return False

while True:
    n, m = map(int, input().split())
    if n == 0 and m == 0:
        break
    edges = []
    for _ in range(m):
        u, v, cost = map(int, input().split())
        edges.append((cost, u, v))
    edges.sort()
    
    dsu = DSU(n)
    total_cost = 0
    connected = 0
    for cost, u, v in edges:
        if dsu.union(u, v):
            total_cost += cost
            connected += 1
            if connected == n - 1:
                break
    print(total_cost)
"""
        ]
    ),

    # -------------------------------------------------------------
    # 5. Quadratic Time: O(N^2)
    # -------------------------------------------------------------
    "p00042": ProblemSpec(
        problem_id="p00042",
        title="A Thief (0/1 Knapsack)",
        algorithmic_paradigm="Dynamic Programming / 2D Grid DP",
        time_complexity="O(N^2)",
        space_complexity="O(N^2)",
        description="Find maximum treasure value fitting within bag capacity W.",
        input_constraints="1 <= n <= 1000, 1 <= W <= 1000",
        sample_solutions=[
            """# Solution 1: Classic 2D dynamic programming
case_num = 1
while True:
    w_limit = int(input())
    if w_limit == 0:
        break
    n = int(input())
    items = []
    for _ in range(n):
        val, weight = map(int, input().split(','))
        items.append((val, weight))
    
    dp = [0] * (w_limit + 1)
    for val, weight in items:
        for w in range(w_limit, weight - 1, -1):
            if dp[w - weight] + val > dp[w]:
                dp[w] = dp[w - weight] + val
    
    max_val = max(dp)
    min_w = dp.index(max_val)
    print(f"Case {case_num}:")
    print(max_val)
    print(min_w)
    case_num += 1
"""
        ]
    ),
    "p00067": ProblemSpec(
        problem_id="p00067",
        title="The Number of Island",
        algorithmic_paradigm="Graph Traversal / 2D Grid Connected Components",
        time_complexity="O(N^2)",
        space_complexity="O(N^2)",
        description="Given a 12x12 binary grid, count the number of distinct connected 4-directional islands.",
        input_constraints="12 lines of 12 binary digits.",
        sample_solutions=[
            """# Solution 1: DFS connected components on 2D grid
import sys

def dfs(grid, r, c):
    grid[r][c] = '0'
    for dr, dc in [(-1,0), (1,0), (0,-1), (0,1)]:
        nr, nc = r + dr, c + dc
        if 0 <= nr < 12 and 0 <= nc < 12 and grid[nr][nc] == '1':
            dfs(grid, nr, nc)

lines = [line.strip() for line in sys.stdin if line.strip()]
for idx in range(0, len(lines), 12):
    grid = [list(lines[idx + r]) for r in range(12)]
    islands = 0
    for r in range(12):
        for c in range(12):
            if grid[r][c] == '1':
                dfs(grid, r, c)
                islands += 1
    print(islands)
"""
        ]
    ),
    "p00118": ProblemSpec(
        problem_id="p00118",
        title="Property Distribution (Flood Fill)",
        algorithmic_paradigm="Graph Traversal / 2D Flood Fill",
        time_complexity="O(N^2)",
        space_complexity="O(N^2)",
        description="Count number of contiguous fruit patches on an H x W orchard grid.",
        input_constraints="1 <= H, W <= 100",
        sample_solutions=[
            """# Solution 1: BFS queue flood fill
from collections import deque

def flood_fill(grid, h, w):
    visited = [[False] * w for _ in range(h)]
    regions = 0
    for r in range(h):
        for c in range(w):
            if not visited[r][c]:
                regions += 1
                fruit = grid[r][c]
                queue = deque([(r, c)])
                visited[r][c] = True
                while queue:
                    cr, cc = queue.popleft()
                    for dr, dc in [(-1,0), (1,0), (0,-1), (0,1)]:
                        nr, nc = cr + dr, cc + dc
                        if 0 <= nr < h and 0 <= nc < w:
                            if not visited[nr][nc] and grid[nr][nc] == fruit:
                                visited[nr][nc] = True
                                queue.append((nr, nc))
    return regions

while True:
    h, w = map(int, input().split())
    if h == 0 and w == 0:
        break
    grid = [input().strip() for _ in range(h)]
    print(flood_fill(grid, h, w))
"""
        ]
    ),

    # -------------------------------------------------------------
    # 6. Cubic Time: O(N^3)
    # -------------------------------------------------------------
    "p00189": ProblemSpec(
        problem_id="p00189",
        title="Convenient Location (Floyd-Warshall)",
        algorithmic_paradigm="Dynamic Programming / All-Pairs Shortest Path",
        time_complexity="O(N^3)",
        space_complexity="O(N^2)",
        description="Find the town that minimizes the sum of shortest path distances to all other towns via Floyd-Warshall.",
        input_constraints="Number of towns up to 10; roads up to 45.",
        sample_solutions=[
            """# Solution 1: Triple nested Floyd-Warshall DP
INF = float('inf')

while True:
    m = int(input())
    if m == 0:
        break
    dist = [[INF] * 10 for _ in range(10)]
    for i in range(10):
        dist[i][i] = 0
    nodes = set()
    for _ in range(m):
        u, v, d = map(int, input().split())
        dist[u][v] = min(dist[u][v], d)
        dist[v][u] = min(dist[v][u], d)
        nodes.add(u)
        nodes.add(v)
    
    # O(V^3) Floyd-Warshall algorithm
    for k in range(10):
        for i in range(10):
            for j in range(10):
                if dist[i][k] + dist[k][j] < dist[i][j]:
                    dist[i][j] = dist[i][k] + dist[k][j]
                    
    best_node = -1
    min_total = INF
    for node in sorted(nodes):
        total = sum(dist[node][other] for other in nodes)
        if total < min_total:
            min_total = total
            best_node = node
    print(f"{best_node} {min_total}")
"""
        ]
    ),
    "p00465": ProblemSpec(
        problem_id="p00465",
        title="Matrix Multiplication",
        algorithmic_paradigm="Algebraic Matrix Operations",
        time_complexity="O(N^3)",
        space_complexity="O(N^2)",
        description="Multiply two N x N matrices with triple nested loops.",
        input_constraints="1 <= n <= 100",
        sample_solutions=[
            """# Solution 1: Classic triple loop matrix multiplication
def multiply_matrices(A, B, n):
    C = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            for k in range(n):
                C[i][j] += A[i][k] * B[k][j]
    return C

n = int(input())
A = [list(map(int, input().split())) for _ in range(n)]
B = [list(map(int, input().split())) for _ in range(n)]
C = multiply_matrices(A, B, n)
for row in C:
    print(*(row))
"""
        ]
    ),

    # -------------------------------------------------------------
    # 7. Exponential Time: O(2^N)
    # -------------------------------------------------------------
    "p00030": ProblemSpec(
        problem_id="p00030",
        title="Sum of Integers (Combinatorial Subset Sum)",
        algorithmic_paradigm="Recursion / Combinatorial Backtracking",
        time_complexity="O(2^N)",
        space_complexity="O(N)",
        description="Count combinations of n distinct digits (0-9) whose sum equals s.",
        input_constraints="1 <= n <= 9, 0 <= s <= 100",
        sample_solutions=[
            """# Solution 1: Recursive DFS branching
def dfs(current_digit, count, current_sum, target_count, target_sum):
    if count == target_count:
        return 1 if current_sum == target_sum else 0
    if current_digit > 9 or current_sum > target_sum:
        return 0
    # Choice 1: Include current digit
    ans = dfs(current_digit + 1, count + 1, current_sum + current_digit, target_count, target_sum)
    # Choice 2: Exclude current digit
    ans += dfs(current_digit + 1, count, current_sum, target_count, target_sum)
    return ans

while True:
    n, s = map(int, input().split())
    if n == 0 and s == 0:
        break
    print(dfs(0, 0, 0, n, s))
"""
        ]
    ),
    "p00033": ProblemSpec(
        problem_id="p00033",
        title="Ball (Recursive Two-Cylinder DFS)",
        algorithmic_paradigm="Backtracking / Binary State Tree",
        time_complexity="O(2^N)",
        space_complexity="O(N)",
        description="Determine whether 10 numbered balls can be dropped into two cylinders so numbers increase in each.",
        input_constraints="Exactly 10 integers.",
        sample_solutions=[
            """# Solution 1: Binary recursive branching
def can_drop(balls, idx, cyl_b, cyl_c):
    if idx == 10:
        return True
    ball = balls[idx]
    # Try cylinder B
    if ball > cyl_b and can_drop(balls, idx + 1, ball, cyl_c):
        return True
    # Try cylinder C
    if ball > cyl_c and can_drop(balls, idx + 1, cyl_b, ball):
        return True
    return False

n = int(input())
for _ in range(n):
    balls = list(map(int, input().split()))
    print("YES" if can_drop(balls, 0, 0, 0) else "NO")
"""
        ]
    ),
    "p00500": ProblemSpec(
        problem_id="p00500",
        title="N-Queens Backtracking",
        algorithmic_paradigm="Exhaustive Search / Backtracking",
        time_complexity="O(2^N)",
        space_complexity="O(N)",
        description="Find all valid non-attacking placements of N queens on an N x N chessboard.",
        input_constraints="1 <= n <= 14",
        sample_solutions=[
            """# Solution 1: Classic recursive backtracking
def solve_n_queens(n):
    cols = set()
    diag1 = set()
    diag2 = set()
    solutions = 0

    def backtrack(row):
        nonlocal solutions
        if row == n:
            solutions += 1
            return
        for col in range(n):
            if col in cols or (row - col) in diag1 or (row + col) in diag2:
                continue
            cols.add(col)
            diag1.add(row - col)
            diag2.add(row + col)
            backtrack(row + 1)
            cols.remove(col)
            diag1.remove(row - col)
            diag2.remove(row + col)

    backtrack(0)
    return solutions

n = int(input())
print(solve_n_queens(n))
"""
        ]
    )
}


def get_problem_spec(problem_id: str) -> Optional[ProblemSpec]:
    """Retrieve problem specification by ID."""
    return PROBLEM_REGISTRY.get(problem_id)


def get_all_problem_ids() -> List[str]:
    """Return all registered problem IDs."""
    return list(PROBLEM_REGISTRY.keys())
