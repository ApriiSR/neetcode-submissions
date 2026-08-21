"""Deterministic input generators for benchmarking NeetCode submissions.

Each problem in `PROBLEMS` maps to:
- `entry`: the Solution method to call, or None for a design problem
  whose submission defines no Solution class at all (minimum-stack,
  lru-cache, implement-prefix-tree, kth-largest-integer-in-a-stream,
  design-word-search-data-structure).
- `scalable`: False for problems benchmark.py can't time. Three have no
  size to vary -- valid-sudoku is a fixed 9x9 board, and reverse-bits and
  sum-of-two-integers are pinned to 32 bits by the submissions
  themselves, which format to exactly 32 binary digits or loop over
  exactly 32 bit positions. The other five -- minimum-stack, lru-cache,
  implement-prefix-tree, kth-largest-integer-in-a-stream and
  design-word-search-data-structure -- are design problems whose
  submissions define no Solution class at all. benchmark.py skips scaling
  curves for all of them, so their `generate` documents the input shape
  rather than feeding a timing run.
- `sizes` / `models`: optional, and set together. A problem whose output is
  exponential in the input size can't walk the default 256..2**20 ladder
  -- one doubling of n squares the work, so its first rung is already
  unreachable -- and can't be described by any of benchmark.py's
  polynomial CANDIDATE_MODELS either. Such a problem carries its own
  ladder and its own extra candidate models instead; see
  EXPONENTIAL_SIZES and EXPONENTIAL_MODELS below (subsets) and
  FIBONACCI_SIZES and FIBONACCI_MODELS (climbing-stairs). Everything else
  inherits benchmark.py's defaults.
- `generate(n, rng)`: returns a positional-args tuple (excluding `self`)
  for `entry`, sized around n. Uses only the passed-in random.Random, so
  it's deterministic for a given seed.
- `build(*description)`: optional, and present only where `entry` takes
  live objects rather than plain data -- the linked-list problems, whose
  entries take ListNode/Node heads, and the binary-tree problems, whose
  entries take TreeNode roots. There `generate` returns a plain,
  comparable *description* (the values a chain or tree would be built
  from) and `build` turns that into the real args tuple. benchmark.py
  calls it fresh before each timed repeat, outside the timer, which is
  what lets in-place solutions be timed at all: every repeat gets an
  untouched structure instead of re-timing the previous repeat's output.
  Keeping `generate` on plain data is also what keeps it comparable,
  which is what the determinism tests check -- node objects compare by
  identity, so a generate that returned them could never be checked that
  way. merge-k-sorted-linked-lists describes its chains as a tuple of
  tuples rather than a list of lists, since its entry takes one list *of*
  heads and the description would otherwise be mistaken for that list;
  the tree problems describe their values as a tuple for the same reason.
- `adversarial(n)`: returns an args tuple built to trigger the worst case
  of the dict/set the entry method is expected to use, or None if the
  problem has no meaningful adversarial-hash story.
- `adversarial_note`: one-line description of the construction, or None.

Integer-keyed adversarial inputs exploit CPython's int hash: for ints,
hash(k) == k (mod sign handling), reduced against the Mersenne prime
P = 2**61 - 1. Every multiple of P hashes to 0 — verified empirically:
hash(0) == hash(P) == hash(2 * P) == 0. `_int_collisions(n)` returns n
such (distinct, ascending) values.

None of the problems here have a dict/set keyed on raw, unbounded
input strings (is-anagram, implement-prefix-tree and
design-word-search-data-structure key on single characters — a 26-symbol
alphabet, too small to stress; anagram-groups
keys on a computed 26-int signature tuple, not the input strings;
string-encode-and-decode doesn't hash at all; the linked-list problems
that hash at all, and level-order-traversal-of-binary-tree submission-1,
key on node objects, whose hashes come from CPython's identity hash, so
the input can't choose them). So there's currently no PYTHONHASHSEED=0-dependent
string-collision generator in use. benchmark.py still forces
PYTHONHASHSEED=0 for the whole run as a forward-looking default, since
str/bytes hashing is otherwise randomized per-process and any future
string-keyed adversarial generator would need it fixed before the
interpreter starts.
"""

import math
import random
import string

MERSENNE61 = 2**61 - 1
RPN_VALUE_LIMIT = 10**6

# The ladder for a problem whose output is exponential in the input size, and
# the only thing that makes such a problem measurable at all: benchmark.py's
# default ladder starts at 256 and doubles, but here one doubling of n squares
# the work, so its very first rung is already out of reach. Stepping by 1 from
# 4 puts 18 points between about 5 microseconds and 1.4 seconds -- a longer
# ladder than the default's 13 -- and stops short of SIZE_CAP_SECONDS on its
# own rather than being truncated by it. A spec opts in via "sizes".
EXPONENTIAL_SIZES = tuple(range(4, 22))

# Paired with EXPONENTIAL_SIZES via "models", because none of benchmark.py's
# CANDIDATE_MODELS can describe an exponential curve: fitting one against them
# picks whichever polynomial is least wrong, which is how subsets used to be
# reported as n^3 log n. Written with a float base on purpose -- 2.0**n raises
# OverflowError past n = 1023 instead of building a multi-megabit integer,
# which is what lets benchmark.model_fits_ladder drop these cheaply if they
# ever meet a ladder they don't belong to.
#
# Worth knowing before reading a winner off the chart: these two separate
# from the polynomials decisively but barely from each other. On the subsets
# ladder the best polynomial reaches r^2 = 0.59 while both of these clear
# 0.995 -- but they sit 0.9955 against 0.9959, because in log space they
# differ only by a log n term. "Exponential, not polynomial" is a strong
# reading of that result; "n * 2^n rather than 2^n" is a weak one.
EXPONENTIAL_MODELS = (
    ("2^n", lambda n: 2.0**n),
    ("n 2^n", lambda n: n * 2.0**n),
)


# climbing-stairs' ladder, and the same idea as EXPONENTIAL_SIZES with a much
# smaller constant factor: the plain recursion there builds no output, so it
# only clears a microsecond around n = 15 and stays under SIZE_CAP_SECONDS
# until about n = 36. Stepping by 1 from 4 to 39 puts that whole measurable
# range on the ladder and lets the per-size cap truncate it wherever the
# machine happens to land.
FIBONACCI_SIZES = tuple(range(4, 40))

# Paired with FIBONACCI_SIZES via "models". The naive climbing-stairs
# recursion recomputes the Fibonacci tree, so its call count is Theta(phi**n),
# not 2**n -- close enough to look exponential on a chart, far enough that
# fitting it against 2**n leaves a residual growing linearly in n. Carried
# alongside EXPONENTIAL_MODELS so the winner can be the right exponential.
GOLDEN_RATIO = (1 + math.sqrt(5)) / 2
FIBONACCI_MODELS = (("phi^n", lambda n: GOLDEN_RATIO**n),) + EXPONENTIAL_MODELS

class ListNode:
    """The singly-linked node NeetCode hands the linked-list problems, whose
    definition their submissions carry as a commented-out header and then
    construct by name. benchmark.py injects this class under that name when
    it loads a submission, so `ListNode(...)` in solution code resolves to
    the same class the `build` hooks below use.
    """

    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next


class Node:
    """The copy-linked-list-with-random-pointer variant: same idea as
    ListNode, plus the extra `random` pointer, and injected the same way.
    """

    def __init__(self, x, next=None, random=None):
        self.val = int(x)
        self.next = next
        self.random = random


class TreeNode:
    """The binary-tree node the tree problems are written against, and what
    their `build` hooks construct. Injected into the loader namespace the
    same way ListNode is, though no tree solution constructs one by name
    today -- what the injection buys is that these submissions no longer
    load *only* because the workflow pins Python 3.14, whose lazy
    annotations (PEP 649) leave the `Optional[TreeNode]` in their signatures
    unevaluated.

    `children` is not part of NeetCode's definition. It's here because
    level-order-traversal-of-binary-tree submission-1 redefines TreeNode in
    its own file, adding that property, and then reads it off whatever nodes
    it is handed -- on NeetCode, nodes of its own redefined class. Supplying
    the property is what lets the shared build hook hand it a tree it can
    walk, and it returns exactly what that submission's own version returns.
    A future submission that redefines TreeNode *incompatibly* (renaming
    `val`, say) would need the loader to hand its own class back to the
    build hook instead; nothing here does that yet.
    """

    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

    @property
    def children(self):
        return [self.left, self.right]


def _int_collisions(n):
    return [k * MERSENNE61 for k in range(n)]


def _chain(values):
    head = None
    for value in reversed(values):
        head = ListNode(value, head)
    return head


def _complete_tree_nodes(values):
    # Level-order fill, so the tree is complete and its depth is
    # floor(log2(n)): shallow enough that the recursive submissions never
    # approach CPython's recursion limit however far up the ladder they get,
    # and balanced, which is the no-early-exit case for the shape-sensitive
    # ones (isBalanced never finds an imbalance to stop on).
    nodes = [TreeNode(value) for value in values]
    for i, node in enumerate(nodes):
        left, right = 2 * i + 1, 2 * i + 2
        node.left = nodes[left] if left < len(nodes) else None
        node.right = nodes[right] if right < len(nodes) else None
    return nodes


def _complete_tree(values):
    nodes = _complete_tree_nodes(values)
    return nodes[0] if nodes else None


def _copy_tree(node):
    if node is None:
        return None
    return TreeNode(node.val, _copy_tree(node.left), _copy_tree(node.right))


def _balanced_bst_nodes(values):
    # `values` ascending; every subtree takes its middle element as its root,
    # so the result is a BST of depth ceil(log2(n)). Returns the root plus
    # the nodes in ascending (in-order) order, so a caller can name two of
    # them by position.
    nodes = [TreeNode(value) for value in values]

    def link(lo, hi):
        if lo > hi:
            return None
        mid = (lo + hi) // 2
        nodes[mid].left = link(lo, mid - 1)
        nodes[mid].right = link(mid + 1, hi)
        return nodes[mid]

    return link(0, len(nodes) - 1), nodes


def _random_word(rng, min_len=3, max_len=10):
    length = rng.randint(min_len, max_len)
    return "".join(rng.choice(string.ascii_lowercase) for _ in range(length))


def _gen_add_two_numbers(n, rng):
    # Digits are stored least-significant first, so the last element is the
    # leading digit and is never 0.
    length = max(1, n)
    l1 = [rng.randint(0, 9) for _ in range(length - 1)] + [rng.randint(1, 9)]
    l2 = [rng.randint(0, 9) for _ in range(length - 1)] + [rng.randint(1, 9)]
    return (l1, l2)


def _build_add_two_numbers(l1, l2):
    return (_chain(l1), _chain(l2))


def _gen_anagram_groups(n, rng):
    return ([_random_word(rng, 3, 8) for _ in range(n)],)


def _adv_anagram_groups(n):
    base = "abcdefghij"
    words = [base[i % len(base):] + base[:i % len(base)] for i in range(n)]
    return (words,)


def _gen_binary_search(n, rng):
    nums = sorted(rng.sample(range(-n * 10, n * 10 + 1), n))
    return (nums, nums[rng.randrange(n)])


def _gen_binary_tree(n, rng):
    # Shared by balanced-binary-tree, binary-tree-diameter,
    # binary-tree-right-side-view, count-good-nodes-in-binary-tree,
    # depth-of-binary-tree, invert-a-binary-tree and
    # level-order-traversal-of-binary-tree: each takes one root and reads
    # nothing but the shape and the values. Returned as a tuple rather than
    # a list so it can't be mistaken for an argument the entry takes.
    return (tuple(rng.randint(-1000, 1000) for _ in range(n)),)


def _build_one_tree(values):
    return (_complete_tree(values),)


def _gen_buy_and_sell_crypto(n, rng):
    return ([rng.randint(1, 1000) for _ in range(n)],)


def _gen_car_fleet(n, rng):
    target = 2 * n + 1
    position = rng.sample(range(target), n)
    speed = [rng.randint(1, 100) for _ in range(n)]
    return (target, position, speed)


def _gen_climbing_stairs(n, rng):
    # The entry takes the stair count itself, so the ladder value is the whole
    # input and there is nothing to sample.
    return (n,)


def _gen_combination_target_sum(n, rng):
    # Every candidate is at least the target and exactly one equals it, so the
    # answer is a single one-element combination and every recursive call dies
    # on its target <= 0 guard. Without that the search is exponential in
    # target / min(candidates) and there is nothing to put on a ladder.
    nums = [n] + rng.sample(range(n + 1, 3 * n), n - 1)
    rng.shuffle(nums)
    return (nums, n)


def _gen_copy_linked_list_with_random_pointer(n, rng):
    values = [rng.randint(-1000, 1000) for _ in range(n)]
    randoms = [rng.randrange(n) if rng.random() < 0.75 else None for _ in range(n)]
    return (values, randoms)


def _build_copy_linked_list_with_random_pointer(values, randoms):
    nodes = [Node(value) for value in values]
    for i, node in enumerate(nodes):
        node.next = nodes[i + 1] if i + 1 < len(nodes) else None
        node.random = None if randoms[i] is None else nodes[randoms[i]]
    return (nodes[0] if nodes else None,)


def _gen_counting_bits(n, rng):
    return (n,)


def _gen_daily_temperatures(n, rng):
    return ([rng.randint(30, 100) for _ in range(n)],)


def _gen_design_word_search_data_structure(n, rng):
    words = [_random_word(rng, 3, 10) for _ in range(max(1, n // 4))]
    ops = []
    for _ in range(n):
        word = rng.choice(words)
        roll = rng.random()
        if roll < 0.4:
            ops.append(("addWord", word))
        elif roll < 0.7:
            ops.append(("search", word))
        else:
            pattern = list(word)
            pattern[rng.randrange(len(pattern))] = "."
            ops.append(("search", "".join(pattern)))
    return (ops,)


def _gen_duplicate_integer(n, rng):
    return (rng.sample(range(n * 4 + 1), n),)


def _adv_duplicate_integer(n):
    return (_int_collisions(n),)


def _gen_eating_bananas(n, rng):
    # Pile sizes grow with n, so the search over eating speeds gains rungs as
    # the piles gain count -- a fixed pile range would have made the number of
    # halvings a constant the search rides for free. The hour budget is 2n,
    # which always admits an answer (eating at max(piles) clears one pile an
    # hour, n hours) while still forcing the speed well above 1.
    size = max(1, n)
    piles = [rng.randint(1, 10 * size) for _ in range(size)]
    return (piles, 2 * size)


def _gen_evaluate_reverse_polish_notation(n, rng):
    # Builds a valid postfix expression by evaluating it as it goes, so "/"
    # is only emitted when the divisor is nonzero and "*" only when the
    # product stays under RPN_VALUE_LIMIT -- otherwise repeated squaring
    # would leave the benchmark timing bignum arithmetic, and int(a / b)
    # would eventually overflow the float conversion.
    operands = max(2, (n + 1) // 2)
    first = rng.randint(1, 100)
    tokens = [str(first)]
    values = [first]
    remaining, ops = operands - 1, operands - 1
    while remaining or ops:
        if ops == 0 or (remaining and (len(values) < 2 or rng.random() < 0.5)):
            value = rng.randint(1, 100)
            tokens.append(str(value))
            values.append(value)
            remaining -= 1
            continue
        b = values.pop()
        a = values.pop()
        choices = ["+", "-"]
        if abs(a) * abs(b) <= RPN_VALUE_LIMIT:
            choices.append("*")
        if b != 0:
            choices.append("/")
        op = rng.choice(choices)
        if op == "+":
            values.append(a + b)
        elif op == "-":
            values.append(a - b)
        elif op == "*":
            values.append(a * b)
        else:
            values.append(int(a / b))
        tokens.append(op)
        ops -= 1
    return (tokens,)


def _gen_find_duplicate_integer(n, rng):
    # NeetCode's shape exactly: n numbers drawn from 1..n-1, so exactly one
    # value repeats. Both occurrences land at random positions rather than
    # being pinned to either end, so a scan that stops at the first repeat
    # walks a fraction of the array that doesn't depend on n.
    size = max(2, n)
    nums = list(range(1, size)) + [rng.randint(1, size - 1)]
    rng.shuffle(nums)
    return (nums,)


def _adv_find_duplicate_integer(n):
    values = _int_collisions(max(2, n) - 1)
    return (values + [values[-1]],)


def _gen_find_minimum_in_rotated_sorted_array(n, rng):
    # Rotated by exactly one position, which puts the minimum at the last
    # index: a left-to-right scan for the one descent has to walk the whole
    # array before it finds it, this problem's no-early-exit case, while a
    # binary search takes the same log2(n) steps at any rotation.
    size = max(2, n)
    values = sorted(rng.sample(range(-size * 10, size * 10 + 1), size))
    return (values[1:] + values[:1],)


def _gen_house_robber(n, rng):
    return ([rng.randint(0, 1000) for _ in range(n)],)


def _gen_implement_prefix_tree(n, rng):
    words = [_random_word(rng, 3, 10) for _ in range(max(1, n // 4))]
    ops = []
    for _ in range(n):
        word = rng.choice(words)
        roll = rng.random()
        if roll < 0.4:
            ops.append(("insert", word))
        elif roll < 0.7:
            ops.append(("search", word))
        else:
            ops.append(("startsWith", word[:rng.randint(1, len(word))]))
    return (ops,)


def _gen_is_anagram(n, rng):
    s = "".join(rng.choice(string.ascii_lowercase) for _ in range(n))
    t = list(s)
    rng.shuffle(t)
    return (s, "".join(t))


def _gen_is_palindrome(n, rng):
    half = "".join(rng.choice(string.ascii_lowercase) for _ in range(n // 2))
    if n % 2:
        return (half + rng.choice(string.ascii_lowercase) + half[::-1],)
    return (half + half[::-1],)


def _gen_k_closest_points_to_origin(n, rng):
    # Coordinates are drawn from a box that grows with n, so squared distances
    # stay spread out and exact ties -- which would drop the sorting
    # submission's tuple comparison through to comparing the coordinate lists
    # themselves -- stay vanishingly rare. k is a quarter of the points rather
    # than a fixed handful, so the answer grows with n too.
    limit = 4 * n
    points = [
        [rng.randint(-limit, limit), rng.randint(-limit, limit)] for _ in range(n)
    ]
    return (points, max(1, n // 4))


def _gen_kth_largest_integer_in_a_stream(n, rng):
    k = max(1, min(3, n))
    initial = [rng.randint(-1000, 1000) for _ in range(max(k, n // 2))]
    ops = [("add", rng.randint(-1000, 1000)) for _ in range(n)]
    return (k, initial, ops)


def _gen_largest_rectangle_in_histogram(n, rng):
    return ([rng.randint(0, 10000) for _ in range(n)],)


def _gen_last_stone_weight(n, rng):
    # Weights are sampled without replacement, so no two stones start equal.
    # That is what pins down the pass count: a pass destroys two stones when
    # the heaviest two match exactly and one otherwise, so distinct starting
    # weights mean about n passes rather than a count that drifts with how
    # often the top of the pile happens to tie. The range scales with n for
    # the same reason -- a fixed one would be almost all ties by the top of
    # the ladder.
    return (rng.sample(range(1, 4 * n + 1), n),)


def _gen_linked_list_cycle_detection(n, rng):
    values = [rng.randint(-1000, 1000) for _ in range(n)]
    return (values, n // 2)


def _build_linked_list_cycle_detection(values, cycle_index):
    nodes = [ListNode(value) for value in values]
    for i in range(len(nodes) - 1):
        nodes[i].next = nodes[i + 1]
    if nodes:
        nodes[-1].next = nodes[cycle_index]
    return (nodes[0] if nodes else None,)


def _gen_longest_consecutive_sequence(n, rng):
    nums = list(range(n))
    rng.shuffle(nums)
    return (nums,)


def _adv_longest_consecutive_sequence(n):
    return (_int_collisions(n),)


def _gen_longest_substring_without_duplicates(n, rng):
    return ("".join(rng.choice(string.ascii_lowercase) for _ in range(n)),)


def _gen_lowest_common_ancestor_in_binary_search_tree(n, rng):
    # p and q are pinned to the quarter and three-quarter in-order positions
    # rather than chosen at random. Random positions make the cost bimodal
    # for a submission that decides ancestry by whole-subtree comparison --
    # O(n) when the root is already the answer, far worse when it has to
    # descend past a wrong branch -- and a ladder built from coin flips
    # measures the flips, not n.
    size = max(4, n)
    values = sorted(rng.sample(range(-size * 10, size * 10 + 1), size))
    return (tuple(values), size // 4, 3 * size // 4)


def _build_lowest_common_ancestor_in_binary_search_tree(values, i, j):
    root, nodes = _balanced_bst_nodes(values)
    return (root, nodes[i], nodes[j])


def _gen_lru_cache(n, rng):
    ops = []
    key_space = max(1, n // 4)
    for _ in range(n):
        key = rng.randint(0, key_space)
        if rng.random() < 0.5:
            ops.append(("get", key))
        else:
            ops.append(("put", key, rng.randint(-1000, 1000)))
    return (max(1, n // 8), ops)


def _gen_max_water_container(n, rng):
    return ([rng.randint(1, 1000) for _ in range(n)],)


def _gen_merge_k_sorted_linked_lists(n, rng):
    # k grows as sqrt(n), so the number of chains and each chain's length
    # both scale with n. Values come from a range that scales with n too,
    # which keeps ties between chain heads rare -- the submissions here
    # short-circuit their head scan on a tie, and a fixed value range would
    # make those ties more frequent as n grew, quietly shrinking the scan.
    total = max(1, n)
    k = max(1, math.isqrt(total))
    base, extra = divmod(total, k)
    groups = []
    for i in range(k):
        length = base + (1 if i < extra else 0)
        groups.append(tuple(sorted(rng.randint(0, 4 * total) for _ in range(length))))
    return (tuple(groups),)


def _build_merge_k_sorted_linked_lists(groups):
    return ([_chain(values) for values in groups],)


def _gen_merge_two_sorted_linked_lists(n, rng):
    half = n // 2
    list1 = sorted(rng.randint(-1000, 1000) for _ in range(half))
    list2 = sorted(rng.randint(-1000, 1000) for _ in range(n - half))
    return (list1, list2)


def _build_merge_two_sorted_linked_lists(list1, list2):
    return (_chain(list1), _chain(list2))


def _gen_min_cost_climbing_stairs(n, rng):
    # Floored at two steps, which is where the recurrence starts; the ladder
    # never goes near that floor.
    return ([rng.randint(0, 1000) for _ in range(max(2, n))],)


def _gen_minimum_stack(n, rng):
    ops = []
    depth = 0
    for _ in range(n):
        if depth and rng.random() < 0.4:
            ops.append(("pop",))
            depth -= 1
        else:
            ops.append(("push", rng.randint(-1000, 1000)))
            depth += 1
    return (ops,)


def _gen_missing_number(n, rng):
    # A shuffled permutation of 0..n-1, so the absent value is always n --
    # the deterministic worst case for a scan that stops at the gap.
    nums = list(range(n))
    rng.shuffle(nums)
    return (nums,)


def _gen_non_cyclical_number(n, rng):
    # n is the input value itself rather than a length, so the generator
    # samples nothing and what the ladder varies is the digit count, about
    # log10(n) of it.
    return (n,)


def _gen_number_of_one_bits(n, rng):
    # n is the width in bits, not a length: the top bit is forced set so the
    # value really is n bits wide and the ladder means what it says.
    bits = max(1, n)
    return (rng.getrandbits(bits) | 1 << (bits - 1),)


def _gen_plus_one(n, rng):
    # n = the number of digits, and the leading one is never 0 so the list is
    # really n digits wide. The rest are uniform over 0..9, which makes a carry
    # die at the last digit nine times in ten and reach the front only about
    # 10**-n of the time.
    return ([rng.randint(1, 9)] + [rng.randint(0, 9) for _ in range(n - 1)],)


def _gen_pow_x_n(n, rng):
    # n is the exponent -- a value rather than a length, so what grows along
    # the ladder is only its bit length. The base is held inside the unit
    # interval (uniform magnitude 0.5..1.0, random sign) so that repeated
    # squaring underflows toward 0.0 instead of raising OverflowError: any
    # base above 1 overflows a float long before the top of the ladder.
    return (rng.choice((-1.0, 1.0)) * rng.uniform(0.5, 1.0), n)


def _gen_products_of_array_discluding_self(n, rng):
    return ([rng.choice((1, -1)) for _ in range(n)],)


def _gen_remove_node_from_end_of_linked_list(n, rng):
    values = [rng.randint(-1000, 1000) for _ in range(n)]
    return (values, rng.randint(1, n) if n else 1)


def _build_remove_node_from_end_of_linked_list(values, position):
    return (_chain(values), position)


def _gen_reorder_linked_list(n, rng):
    return ([rng.randint(-1000, 1000) for _ in range(n)],)


def _gen_reverse_a_linked_list(n, rng):
    return ([rng.randint(-1000, 1000) for _ in range(n)],)


def _build_one_list(values):
    # Shared by reorder-linked-list and reverse-a-linked-list: both entries
    # take a single head and both rewire it in place.
    return (_chain(values),)


def _gen_reverse_bits(n, rng):
    return (rng.getrandbits(32),)


def _gen_reverse_integer(n, rng):
    return (rng.randint(-(2**31), 2**31 - 1),)


def _gen_same_binary_tree(n, rng):
    return (tuple(rng.randint(-1000, 1000) for _ in range(n)),)


def _build_same_binary_tree(values):
    # Two structurally identical trees over the same values, so the
    # comparison never short-circuits and always walks all n nodes twice.
    return (_complete_tree(values), _complete_tree(values))


def _gen_search_2d_matrix(n, rng):
    # rows and columns both grow like sqrt(n), so neither dimension is a
    # constant the search can be linear in for free.
    total = max(1, n)
    rows = max(1, math.isqrt(total))
    cols = max(1, total // rows)
    values = sorted(rng.sample(range(-total * 10, total * 10 + 1), rows * cols))
    matrix = [values[r * cols:(r + 1) * cols] for r in range(rows)]
    return (matrix, values[rng.randrange(rows * cols)])


def _gen_single_number(n, rng):
    # (n - 1) // 2 pairs plus one unpaired value, sampled without replacement
    # so no pair shares a value with another, then shuffled so the pairs are
    # scattered rather than adjacent.
    pairs = max(0, (n - 1) // 2)
    values = rng.sample(range(-4 * n, 4 * n), pairs + 1)
    nums = values[:pairs] * 2 + values[pairs:]
    rng.shuffle(nums)
    return (nums,)


def _gen_string_encode_and_decode(n, rng):
    return ([_random_word(rng, 3, 12) for _ in range(n)],)


def _gen_subsets(n, rng):
    # n is the array length itself, walked one step at a time over
    # EXPONENTIAL_SIZES rather than doubled. Values are sampled without
    # replacement from a range that scales with n, since the problem
    # guarantees them distinct.
    size = max(1, n)
    return (rng.sample(range(-2 * size, 2 * size + 1), size),)


def _gen_subtree_of_a_binary_tree(n, rng):
    # The node a pre-order search reaches last: from the root, take the right
    # child while there is one and otherwise the left. Copying that node's
    # subtree keeps the answer True while still forcing a full traversal, so
    # the timing is the traversal rather than wherever a randomly chosen
    # target happened to sit.
    size = max(1, n)
    values = list(range(size))
    rng.shuffle(values)
    index = 0
    while True:
        left, right = 2 * index + 1, 2 * index + 2
        if right < size:
            index = right
        elif left < size:
            index = left
        else:
            break
    return (tuple(values), index)


def _build_subtree_of_a_binary_tree(values, index):
    nodes = _complete_tree_nodes(values)
    return (nodes[0], _copy_tree(nodes[index]))


def _gen_sum_of_two_integers(n, rng):
    return (rng.randint(-1000, 1000), rng.randint(-1000, 1000))


def _gen_three_integer_sum(n, rng):
    # Values are sampled without replacement from a range that grows like
    # n**2, which keeps the number of zero-sum triples linear in n (25 of
    # them at n = 512). A range that grew only linearly with n would leave
    # quadratically many, and the submission's dedup check scans the whole
    # result list per hit, so the answer set -- not the search -- would
    # dominate the run.
    span = max(1, n * n)
    return (rng.sample(range(-span, span + 1), n),)


def _gen_top_k_elements_in_list(n, rng):
    k = min(5, n) if n else 0
    return ([rng.randint(0, max(1, n // 4)) for _ in range(n)], k)


def _adv_top_k_elements_in_list(n):
    k = min(5, n) if n else 0
    return (_int_collisions(n), k)


def _gen_trapping_rain_water(n, rng):
    return ([rng.randint(0, 1000) for _ in range(n)],)


def _gen_two_integer_sum(n, rng):
    nums = rng.sample(range(-n * 10, n * 10 + 1), n)
    i, j = rng.sample(range(n), 2)
    target = nums[i] + nums[j]
    return (nums, target)


def _adv_two_integer_sum(n):
    nums = _int_collisions(n)
    target = nums[0] + nums[1]
    return (nums, target)


def _gen_two_integer_sum_ii(n, rng):
    nums = sorted(rng.sample(range(-n * 10, n * 10 + 1), n))
    i, j = sorted(rng.sample(range(n), 2))
    target = nums[i] + nums[j]
    return (nums, target)


def _adv_two_integer_sum_ii(n):
    nums = _int_collisions(n)
    target = nums[0] + nums[1]
    return (nums, target)


def _gen_validate_parentheses(n, rng):
    pairs = [("(", ")"), ("[", "]"), ("{", "}")]
    out = []
    stack = []
    remaining = n // 2
    while remaining or stack:
        if remaining and (not stack or rng.random() < 0.5):
            op, cl = rng.choice(pairs)
            out.append(op)
            stack.append(cl)
            remaining -= 1
        else:
            out.append(stack.pop())
    return ("".join(out),)


def _gen_valid_sudoku(n, rng):
    board = [["." for _ in range(9)] for _ in range(9)]
    for box_r in range(3):
        for box_c in range(3):
            digits = [str(d) for d in range(1, 10)]
            rng.shuffle(digits)
            keep = digits[:rng.randint(3, 9)]
            cells = [(r, c) for r in range(3) for c in range(3)]
            rng.shuffle(cells)
            for (r, c), d in zip(cells, keep):
                board[box_r * 3 + r][box_c * 3 + c] = d
    return (board,)


PROBLEMS = {
    "add-two-numbers": {
        "entry": "addTwoNumbers",
        "scalable": True,
        "generate": _gen_add_two_numbers,
        "build": _build_add_two_numbers,
        "adversarial": None,
        "adversarial_note": None,
        "scaling_note": "n = number of digits in each addend; both linked lists are exactly n digits long (stored least-significant first, leading digit never 0), so the two addends always have equal length and the sum has n or n+1 digits. Digits are drawn uniformly from 0..9, so digit values are independent of n and carries propagate at the usual ~50% rate rather than being forced",
        "generalized_note": "uncapped: both lists any length. The digit alphabet is part of the problem, not a cap — still 0-9, least-significant first, with no leading zero. Per-digit arithmetic is unit cost, but a solution that decodes a whole list into a single integer is then operating on n-digit numbers, whose arithmetic is not unit cost, and one that recurses once per node needs stack depth proportional to n.",
    },
    "anagram-groups": {
        "entry": "groupAnagrams",
        "scalable": True,
        "generate": _gen_anagram_groups,
        "adversarial": _adv_anagram_groups,
        "adversarial_note": "every word is a rotation of the same 10-letter multiset, so all n words share one anagram signature (one exact dict key, not merely one hash bucket) and land in a single group; empirically this does NOT degrade to O(n^2) the way distinct-but-colliding keys do, since CPython dicts resolve a repeated exact key by direct slot lookup rather than probing a chain — see README",
        "scaling_note": "n = number of words; each word's length is drawn from a fixed small range (3-8 chars) independent of n, so total character volume scales linearly with n",
        "generalized_note": "uncapped: any number of words, each of any length. The alphabet is part of the problem, not a cap \u2014 words stay lowercase English letters, so the 26-letter signature space does not grow with n.",
    },
    "balanced-binary-tree": {
        "entry": "isBalanced",
        "scalable": True,
        "generate": _gen_binary_tree,
        "build": _build_one_tree,
        "adversarial": None,
        "adversarial_note": None,
        "scaling_note": "n = number of nodes; the tree is filled level order, so it is complete and its depth is floor(log2(n)) -- always balanced, which is this problem's no-early-exit case: a submission never finds an imbalance to stop on and so recurses over every node. Because the depth stays logarithmic in n, a recursive submission never approaches CPython's stack limit anywhere on the ladder, and a submission that recomputes a subtree's height at every node does work proportional to n * depth rather than the n a single bottom-up pass would take. Node values are drawn uniformly from the fixed range -1000..1000, independent of n; nothing in this problem reads them",
        "generalized_note": "uncapped: any number of nodes, values any ints, and any shape -- a tree may be arbitrarily skewed rather than complete, so both its height and a recursive solution's stack depth can be linear in n, and the stated node-count maximum is a cap that goes. The balance condition is part of the problem, not a cap: every node's two subtrees must still differ in height by at most 1.",
    },
    "binary-search": {
        "entry": "search",
        "scalable": True,
        "generate": _gen_binary_search,
        "adversarial": None,
        "adversarial_note": None,
        "scaling_note": "n = array length; values are sampled without replacement from a range that scales with n (-10n..10n) and then sorted ascending, and the target is always one of the n values (never absent), so every run finds its target",
        "generalized_note": "uncapped: any array length, values any ints. Sortedness and uniqueness are part of the problem, not caps, and the target stays one of the values.",
    },
    "binary-tree-diameter": {
        "entry": "diameterOfBinaryTree",
        "scalable": True,
        "generate": _gen_binary_tree,
        "build": _build_one_tree,
        "adversarial": None,
        "adversarial_note": None,
        "scaling_note": "n = number of nodes; the tree is filled level order, so it is complete, its depth is floor(log2(n)) and every node's two subtrees are near-equal in size -- meaning the longest path runs through the root and a submission that recomputes subtree height at each node does work proportional to n * depth, where a single bottom-up pass stays proportional to n. Node values are drawn uniformly from the fixed range -1000..1000, independent of n; the diameter depends only on the shape",
        "generalized_note": "uncapped: any number of nodes, values any ints, any shape. The diameter definition is part of the problem, not a cap -- still the longest path between any two nodes, counted in edges, and it need not pass through the root. A skewed tree makes both the height and a recursive solution's stack depth linear in n.",
    },
    "binary-tree-right-side-view": {
        "entry": "rightSideView",
        "scalable": True,
        "generate": _gen_binary_tree,
        "build": _build_one_tree,
        "adversarial": None,
        "adversarial_note": None,
        "scaling_note": "n = number of nodes; the tree is filled level order, so it is complete, it has floor(log2(n)) + 1 levels and each level is twice the width of the one above -- the widest level alone holds about half of all n nodes. The answer is one value per level, so it stays logarithmic in n while the traversal still has to reach every node: the measured time tracks the node count, not the output length. A submission that materialises each full level before taking its last element allocates on the order of n values of scratch on the way to that log-sized answer, and one whose per-level work is quadratic in the level width (popping from the front of a Python list, say) pays about (n/2)**2 on the widest level alone, where a deque stays linear. Node values are drawn uniformly from the fixed range -1000..1000, independent of n, and nothing reads them beyond copying the rightmost of each level into the output",
        "generalized_note": "uncapped: any number of nodes, values any ints, any shape. The output contract is part of the problem, not a cap -- still the first node visible at each depth looking from the right, top to bottom. A skewed tree inverts this input's shape: n levels of one node each rather than log n levels whose widths double, so the answer becomes as long as the tree is tall and a recursive solution needs stack depth proportional to n.",
    },
    "buy-and-sell-crypto": {
        "entry": "maxProfit",
        "scalable": True,
        "generate": _gen_buy_and_sell_crypto,
        "adversarial": None,
        "adversarial_note": None,
        "scaling_note": "n = array length; each price is drawn uniformly from the fixed range 1..1000, independent of n",
        "generalized_note": "uncapped: any array length, prices any non-negative ints. Chronological order is part of the problem.",
    },
    "car-fleet": {
        "entry": "carFleet",
        "scalable": True,
        "generate": _gen_car_fleet,
        "adversarial": None,
        "adversarial_note": None,
        "scaling_note": "n = number of cars; target (the finish line) is set to 2n+1 and the n distinct positions are sampled without replacement from range(target), so both the target and the position range scale linearly with n -- meaning the counting-array submissions, which allocate a list of length target, stay O(n) rather than being dominated by a fixed target; speeds are drawn from the fixed range 1..100, independent of n",
        "generalized_note": "uncapped: any number of cars, and target/speed/position any positive ints. The invariants stay: positions are distinct and all below target, speeds are positive. Note this makes target independent of n, so a solution that allocates an array of size target is no longer O(n).",
    },
    "climbing-stairs": {
        "entry": "climbStairs",
        "scalable": True,
        "generate": _gen_climbing_stairs,
        "sizes": FIBONACCI_SIZES,
        "models": FIBONACCI_MODELS,
        "adversarial": None,
        "adversarial_note": None,
        "scaling_note": "n = the number of stairs, and it is the entire input -- the entry takes one integer, so the size being varied is that integer's value rather than any length, and the generator samples nothing. The ladder is this problem's own: n steps by 1 from 4 to 39 rather than doubling from 256, because the plain recursion recomputes the Fibonacci tree and makes about 2 * fib(n+1) calls, which passes a second somewhere near n = 36 and would never return at the default ladder's first rung. Two things follow for reading the numbers. The reported log-log slope is meaningless here -- it fits log(time) against log(n), and an exponential curve is not a line in those coordinates -- so read best_fit, which is chosen from phi^n, 2^n and n * 2^n alongside the usual polynomials; phi^n (the golden ratio, about 1.618) is the shape the naive recursion actually has, and 2^n overstates it. And the ladder that makes that submission measurable is far too short for the other two: the memoised one runs 0.25 to 1.5 microseconds across the whole of it and the matrix-power one 2 to 5, so their curves are as much timer resolution and call overhead as they are complexity, and their fits should be read that way",
        "generalized_note": "uncapped: any number of stairs. The step set is part of the problem, not a cap -- a move is still 1 or 2 stairs, so the answer is fib(n+1). Note that n is a value rather than a length, so it costs only about log2(n) bits to write down: an O(n) solution is already exponential in the size of its input, and the answer itself is an O(n)-bit number, which is a floor on any solution that prints it.",
    },
    "combination-target-sum": {
        "entry": "combinationSum",
        "scalable": True,
        "generate": _gen_combination_target_sum,
        "adversarial": None,
        "adversarial_note": None,
        "scaling_note": "n = number of candidate values, and it is the target as well: the candidates are n distinct values, exactly one of them equal to the target n and the rest drawn from n+1..3n-1, then shuffled. That no candidate is smaller than the target is the whole design. It means every recursive call the submission makes returns immediately on its `target <= 0` guard, which is what keeps the run finite and pins the answer to one combination, [n]; on candidates spread down toward 1 the search is instead exponential in target / min(candidates) and there is nothing to put on a ladder at all. What is left to measure is the cost the submissions pay per candidate regardless of whether the branch survives: they recurse on the slice nums[i:] rather than on an index, so one pass over n candidates copies about n**2 / 2 elements. That is the quadratic the ladder shows. It is C-level copying rather than interpreted work, so the constant is small and the curve only leaves a millisecond around n = 2000 -- but it is quadratic, and it is the slicing rather than the search",
        "generalized_note": "uncapped: any number of candidates, values any positive ints, target any positive int. Distinctness and the reuse rule are part of the problem, not caps -- the candidates are distinct, each may be used any number of times, and every combination appears exactly once as a multiset, in any order. Note the answer size is not bounded by the candidate count: with candidates small relative to the target the number of combinations, and so any solution's running time, grows exponentially in target / min(candidates).",
    },
    "copy-linked-list-with-random-pointer": {
        "entry": "copyRandomList",
        "scalable": True,
        "generate": _gen_copy_linked_list_with_random_pointer,
        "build": _build_copy_linked_list_with_random_pointer,
        "adversarial": None,
        "adversarial_note": None,
        "scaling_note": "n = number of nodes; each node's random pointer targets a uniformly chosen node ~75% of the time and is null the other ~25%, so the number of random pointers to follow scales linearly with n and their targets are spread across the whole list rather than clustered. Values are drawn from the fixed range -1000..1000, independent of n",
        "generalized_note": "uncapped: any number of nodes, values any ints. The pointer structure is part of the problem, not a cap — each random pointer still targets some node in the list or null. The dict here is keyed on node objects, so its hashes come from CPython's identity hash and the input cannot choose them.",
    },
    "count-good-nodes-in-binary-tree": {
        "entry": "goodNodes",
        "scalable": True,
        "generate": _gen_binary_tree,
        "build": _build_one_tree,
        "adversarial": None,
        "adversarial_note": None,
        "scaling_note": "n = number of nodes; the tree is filled level order, so it is complete and its depth is floor(log2(n)) -- shallow enough that a recursive submission never approaches CPython's stack limit anywhere on the ladder. Every node is visited whatever the values are: the running maximum of the root-to-node path is carried downward and nothing prunes, so the work is proportional to n and independent of how many nodes turn out to be good. Node values are drawn uniformly from the fixed range -1000..1000, independent of n, so a node at depth d is good with probability about 1/(d+1) and the expected number of good nodes is on the order of n / log2(n) -- the answer grows more slowly than n while the traversal does not",
        "generalized_note": "uncapped: any number of nodes, values any ints, any shape. The definition is part of the problem, not a cap -- a node is good when no node on the path from the root down to it holds a strictly greater value, so the root is always good. A skewed tree makes both the depth and a recursive solution's stack linear in n, and makes the good-node count as large as n when values ascend along that single path.",
    },
    "counting-bits": {
        "entry": "countBits",
        "scalable": True,
        "generate": _gen_counting_bits,
        "adversarial": None,
        "adversarial_note": None,
        "scaling_note": "n = the input integer, which is the entire input and also fixes the output length: the answer is a list of n + 1 popcounts, one for each integer from 0 through n. The generator samples nothing. The submission recomputes each count with its own divide-by-2 loop, so the work is the sum of bit_length(i) over i <= n, about n * log2(n) -- the log factor here is the integers' width, not a data structure",
        "generalized_note": "uncapped: any n. The output contract is part of the problem, not a cap -- still one popcount per integer from 0 through n, in order, so the output alone is n + 1 numbers and no solution beats linear in n. Note that n is a value rather than a length, so linear in n is already exponential in the size of the input, which is about log2(n) bits.",
    },
    "daily-temperatures": {
        "entry": "dailyTemperatures",
        "scalable": True,
        "generate": _gen_daily_temperatures,
        "adversarial": None,
        "adversarial_note": None,
        "scaling_note": "n = array length; each temperature is drawn uniformly from the fixed range 30..100, independent of n, so long monotonic runs (the monotonic stack's worst case) do not grow with n",
        "generalized_note": "uncapped: any array length, temperatures any ints. Nothing bounds the run lengths a monotonic stack can face.",
    },
    "depth-of-binary-tree": {
        "entry": "maxDepth",
        "scalable": True,
        "generate": _gen_binary_tree,
        "build": _build_one_tree,
        "adversarial": None,
        "adversarial_note": None,
        "scaling_note": "n = number of nodes; the tree is filled level order, so it is complete and its depth is floor(log2(n)) -- the answer grows only logarithmically while the traversal still has to visit all n nodes, so the measured time tracks the node count and not the answer. Node values are drawn uniformly from the fixed range -1000..1000, independent of n; nothing in this problem reads them",
        "generalized_note": "uncapped: any number of nodes, values any ints, any shape. Depth is still counted in nodes along the longest root-to-leaf path. A skewed tree makes that depth linear in n, so a recursive solution's stack use is linear too.",
    },
    "design-word-search-data-structure": {
        "entry": None,
        "scalable": False,
        "generate": _gen_design_word_search_data_structure,
        "adversarial": None,
        "adversarial_note": None,
        "scaling_note": "not benchmarked: this is a design problem -- the submission defines a WordDictionary class with addWord/search rather than a Solution class with one entry method, so benchmark.py's loader has nothing to drive; the generator builds an n-operation sequence (~40% addWord, ~30% exact search, ~30% search with one character replaced by the '.' wildcard) drawn from a pool of n // 4 lowercase words of 3-10 characters, so words repeat and both hits and misses are common, purely as documentation of the input shape",
        "generalized_note": "uncapped: any number of operations, words of any length. The alphabet and the wildcard are part of the problem, not caps -- words stay lowercase English letters, so each node's child count is bounded by 26 and does not grow with n, and '.' in a search pattern still matches any single letter. Note the wildcard is what makes search more than a walk down one path: a pattern with k dots can force a branch across every child at each of those positions, so its cost is not bounded by the pattern length alone.",
    },
    "duplicate-integer": {
        "entry": "hasDuplicate",
        "scalable": True,
        "generate": _gen_duplicate_integer,
        "adversarial": _adv_duplicate_integer,
        "adversarial_note": "n distinct multiples of 2**61-1, all hashing to 0, so every dict insert/lookup collides",
        "scaling_note": "n = array length; values are sampled without replacement from range(4n+1), so the value range scales with n too",
        "generalized_note": "uncapped: any array length, values any ints \u2014 including values an adversary picks to collide in CPython's hash (multiples of 2**61-1), which the stated -10^9..10^9 range would have made impossible.",
    },
    "eating-bananas": {
        "entry": "minEatingSpeed",
        "scalable": True,
        "generate": _gen_eating_bananas,
        "adversarial": None,
        "adversarial_note": None,
        "scaling_note": "n = number of piles; pile sizes are drawn uniformly from 1..10n, so the range of candidate eating speeds grows with n too and a binary search over it takes about log2(10n) steps rather than a constant number. Each feasibility check sums a ceiling division over all n piles, so a binary-searching submission does about n log n work, while a submission that walks candidate speeds one at a time is linear in max(piles) as well as in n. The hour budget is 2n, which always leaves an answer -- eating at max(piles) clears one pile an hour, n hours, inside the budget -- and puts the answer around 3-4 times n, well away from both ends of the search range. Piles are drawn independently and may repeat; nothing in this problem requires them distinct",
        "generalized_note": "uncapped: any number of piles, pile sizes and hour budget any positive ints. That the budget is at least the number of piles is part of the problem, not a cap -- Koko eats from only one pile per hour, so a smaller budget admits no speed at all.",
    },
    "evaluate-reverse-polish-notation": {
        "entry": "evalRPN",
        "scalable": True,
        "generate": _gen_evaluate_reverse_polish_notation,
        "adversarial": None,
        "adversarial_note": None,
        "scaling_note": "n = number of tokens; the expression is a valid postfix expression with (n+1)//2 operands and one fewer operator, so token count scales linearly with n; operands are drawn from the fixed range 1..100 and operators are picked so intermediate values stay bounded by ~10**6 (no bignum growth), meaning per-token arithmetic cost is constant and independent of n",
        "generalized_note": "uncapped: any number of tokens, operands any ints. The token grammar is part of the problem, not a cap \u2014 still the four operators over a valid postfix expression. Intermediate values may exceed machine words; arithmetic is still counted as unit cost.",
    },
    "find-duplicate-integer": {
        "entry": "findDuplicate",
        "scalable": True,
        "generate": _gen_find_duplicate_integer,
        "adversarial": _adv_find_duplicate_integer,
        "adversarial_note": "n-1 distinct multiples of 2**61-1, all hashing to 0, followed by a repeat of the last of them -- so a set-based scan does n-1 colliding inserts before it reaches the duplicate; the repeat itself is an exact-key hit, which CPython resolves by direct slot lookup, so the degradation comes entirely from the distinct colliding keys ahead of it",
        "scaling_note": "n = array length; the array is a shuffled permutation of 1..n-1 plus one extra copy of a uniformly chosen value from that same range, so exactly one value repeats and the value range scales with n. The two occurrences sit at uniformly random positions rather than being pinned to either end: a scan that stops at the first repeat therefore reaches it about two thirds of the way through on average, and an all-pairs scan reaches the earlier occurrence about a third of the way in, so both do a fixed fraction of their worst case rather than a fraction that shrinks with n",
        "generalized_note": "uncapped: any array length, values any ints — including values an adversary picks to collide in CPython's hash (multiples of 2**61-1). The precondition that some value repeats is part of the problem and stays, but the 1..n-1 value range is a cap, so it goes: that range is what lets the array be read as a function from indices to indices, so a cycle-detection or index-marking solution is correct only under it.",
    },
    "find-minimum-in-rotated-sorted-array": {
        "entry": "findMin",
        "scalable": True,
        "generate": _gen_find_minimum_in_rotated_sorted_array,
        "adversarial": None,
        "adversarial_note": None,
        "scaling_note": "n = array length; n distinct values are sampled without replacement from -10n..10n, sorted ascending, then rotated left by exactly one position, which puts the minimum at the last index. That is this problem's no-early-exit case for a left-to-right scan looking for the single descent: it has to walk the whole array before reaching it, so such a submission comes out linear. A binary search is unaffected by where the rotation falls -- it halves the interval a fixed log2(n) times either way -- so expect a flat, near-zero slope from one. Note the whole ladder runs in microseconds for a binary search, close to the clock's resolution, so read the flatness rather than the r^2",
        "generalized_note": "uncapped: any array length, values any ints. Sortedness, distinctness and the rotation are part of the problem, not caps -- the array is still a strictly increasing sequence rotated some number of positions. The rotation may be zero, which leaves the array plainly sorted and its minimum first; the rotation of one used here is the opposite extreme.",
    },
    "house-robber": {
        "entry": "rob",
        "scalable": True,
        "generate": _gen_house_robber,
        "adversarial": None,
        "adversarial_note": None,
        "scaling_note": "n = array length; each value is drawn uniformly from the fixed range 0..1000, independent of n. The values do not move the measurement at all -- the submission is a single left-to-right pass doing one max per index, and both sides of that comparison cost the same -- so the range is arbitrary and only the length matters",
        "generalized_note": "uncapped: any array length, values any non-negative ints. The adjacency rule is part of the problem, not a cap -- no two chosen houses may be adjacent in the array, and the answer is the largest total that respects that.",
    },
    "implement-prefix-tree": {
        "entry": None,
        "scalable": False,
        "generate": _gen_implement_prefix_tree,
        "adversarial": None,
        "adversarial_note": None,
        "scaling_note": "not benchmarked: this is a design problem -- the submission defines a PrefixTree class with insert/search/startsWith rather than a Solution class with one entry method, so benchmark.py's loader has nothing to drive; the generator builds an n-operation sequence (~40% insert, ~30% search, ~30% startsWith on a prefix of a pooled word) drawn from a pool of n // 4 lowercase words of 3-10 characters, so words repeat and both hits and misses are common, purely as documentation of the input shape",
        "generalized_note": "uncapped: any number of operations, words of any length. The alphabet is part of the problem, not a cap -- words stay lowercase English letters, so each node's child count is bounded by 26 and does not grow with n. The contract stays too: search matches a whole inserted word, startsWith matches any inserted word's prefix.",
    },
    "invert-a-binary-tree": {
        "entry": "invertTree",
        "scalable": True,
        "generate": _gen_binary_tree,
        "build": _build_one_tree,
        "adversarial": None,
        "adversarial_note": None,
        "scaling_note": "n = number of nodes; the tree is filled level order, so it is complete, its depth is floor(log2(n)) and the inversion touches every node exactly once. Node values are drawn uniformly from the fixed range -1000..1000, independent of n, and the work depends only on the node count. The submission rewires the tree in place, so the `build` hook hands each timed repeat a freshly built, un-inverted tree -- inverting twice costs the same as inverting once here, but every repeat still measures the stated input rather than the previous repeat's output",
        "generalized_note": "uncapped: any number of nodes, values any ints, any shape. The inversion is part of the problem, not a cap -- every node's two children are still swapped, and the tree is still rewired rather than rebuilt. A skewed tree makes recursion depth linear in n.",
    },
    "is-anagram": {
        "entry": "isAnagram",
        "scalable": True,
        "generate": _gen_is_anagram,
        "adversarial": None,
        "adversarial_note": None,
        "scaling_note": "n = length of each input string; both strings are always the same length n",
        "generalized_note": "uncapped: both strings any length. The alphabet is part of the problem, not a cap \u2014 still lowercase English letters, so the 26-letter counting space does not grow with n.",
    },
    "is-palindrome": {
        "entry": "isPalindrome",
        "scalable": True,
        "generate": _gen_is_palindrome,
        "adversarial": None,
        "adversarial_note": None,
        "scaling_note": "n = length of the input string",
        "generalized_note": "uncapped: any string length. The character set is part of the problem, not a cap \u2014 still printable ASCII.",
    },
    "k-closest-points-to-origin": {
        "entry": "kClosest",
        "scalable": True,
        "generate": _gen_k_closest_points_to_origin,
        "adversarial": None,
        "adversarial_note": None,
        "scaling_note": "n = number of points, and k scales with it: k is n // 4, so the answer is a quarter of the input rather than a fixed handful and its length grows with n. Both coordinates of every point are drawn uniformly and independently from -4n..4n, so the box grows with n and exact ties in distance stay vanishingly rare -- which matters because the sorting submission keys on the tuple (distance, point), and a tie would drop through to comparing the coordinate lists themselves. Points are not required distinct and may repeat. Because k grows with n, a full sort at n log n and a size-k heap at n log k are separated only by the log(4) between them here, so those two shapes are not distinguishable on this ladder; a linear-average selection (quickselect) is the one that would show up as a different curve",
        "generalized_note": "uncapped: any number of points, coordinates any ints, k anywhere from 1 to the number of points. The output contract is part of the problem, not a cap -- still exactly k points, closest to the origin by Euclidean distance, in any order, with ties broken arbitrarily. Note that k need not grow with n: at a fixed k a heap solution is n log k, effectively linear, where a full sort stays n log n.",
    },
    "kth-largest-integer-in-a-stream": {
        "entry": None,
        "scalable": False,
        "generate": _gen_kth_largest_integer_in_a_stream,
        "adversarial": None,
        "adversarial_note": None,
        "scaling_note": "not benchmarked: this is a design problem -- the submission defines a KthLargest class constructed with (k, nums) and then driven by add() rather than a Solution class with one entry method, so benchmark.py's loader has nothing to drive; the generator builds k = min(3, n), an initial array of max(k, n // 2) values and a stream of n add() calls, all values drawn uniformly from the fixed range -1000..1000, purely as documentation of the input shape",
        "generalized_note": "uncapped: any number of add() calls, any initial array length, values any ints. The contract is part of the problem, not a cap -- add still returns the kth largest value in the stream so far, counting duplicates as separate values, k stays at least 1, and the initial array still holds at least k - 1 values so that the first add() has an answer.",
    },
    "largest-rectangle-in-histogram": {
        "entry": "largestRectangleArea",
        "scalable": True,
        "generate": _gen_largest_rectangle_in_histogram,
        "adversarial": None,
        "adversarial_note": None,
        "scaling_note": "n = number of bars; each height is drawn uniformly from 0..10000, NeetCode's full stated range for heights[i], independent of n -- so a submission that sweeps every height value from 0 to max(heights) does a bounded (<=10001) number of linear passes: still O(n) in n, but with a constant large enough that the size ladder hits its per-size time cap around n = 2**13 to 2**14, depending on machine speed",
        "generalized_note": "uncapped: any number of bars, heights any non-negative ints. Note the stated 0..10000 height cap is what makes a height-indexed sweep O(n); without it such a solution scales with the largest height instead.",
    },
    "last-stone-weight": {
        "entry": "lastStoneWeight",
        "scalable": True,
        "generate": _gen_last_stone_weight,
        "adversarial": None,
        "adversarial_note": None,
        "scaling_note": "n = number of stones; weights are sampled without replacement from range(1, 4n+1), so the weight range scales with n and no two stones start equal. That fixes the number of smashes at about n: a pass destroys both of the two heaviest stones only when they match exactly, and otherwise destroys one and writes their difference back, so with distinct weights the pile shrinks by one per pass. Ties can still appear later among the differences, which are small (the gap between adjacent order statistics averages about 4) and re-enter at the bottom of the order, so the top of the pile stays distinct until late. Each pass therefore sees a list one shorter than the last, and a submission that re-sorts the whole pile every pass does quadratic total work -- the per-pass sort itself stays close to linear rather than n log n, since Timsort finds one long ascending run plus the single element the previous pass disturbed -- measured at about 20x cheaper than sorting the same list shuffled -- while a heap-based submission pays n to heapify and then about n pops and pushes at log n each",
        "generalized_note": "uncapped: any number of stones, weights any positive ints. The smash rule is part of the problem, not a cap: the two heaviest stones are destroyed outright if equal, and otherwise the heavier is replaced by the difference of the two, repeating until at most one stone is left -- the answer is that stone's weight, or 0 if none is.",
    },
    "level-order-traversal-of-binary-tree": {
        "entry": "levelOrder",
        "scalable": True,
        "generate": _gen_binary_tree,
        "build": _build_one_tree,
        "adversarial": None,
        "adversarial_note": None,
        "scaling_note": "n = number of nodes; the tree is filled level order, so it is complete, it has floor(log2(n)) + 1 levels and each level is twice the width of the one above -- meaning the single widest level holds about half of all n nodes. That is the load-bearing fact here: a submission whose per-level work is quadratic in that level's width (popping from the front of a Python list, or checking membership against a collection it is still filling) pays on the order of (n/2)**2 on that one level alone and is quadratic overall, where one that pops from a deque stays linear. The quadratic term only overtakes the linear one partway up the ladder, so a least-squares fit over the whole range reads well below 2 -- the tail quadrupling per doubling is the real signal. Values are drawn uniformly from the fixed range -1000..1000, independent of n, and nothing reads them beyond copying them into the output",
        "generalized_note": "uncapped: any number of nodes, values any ints, any shape. The output contract is part of the problem, not a cap -- still one list per level, top to bottom, each holding that level's values left to right. Note that a skewed tree inverts the shape this input has: n levels of one node each rather than log n levels whose widths double, so a solution whose per-level cost is quadratic in the level width looks linear there and a recursive one needs stack depth proportional to n.",
    },
    "linked-list-cycle-detection": {
        "entry": "hasCycle",
        "scalable": True,
        "generate": _gen_linked_list_cycle_detection,
        "build": _build_linked_list_cycle_detection,
        "adversarial": None,
        "adversarial_note": None,
        "scaling_note": "n = number of nodes; the tail always links back to the node at index n // 2, so every generated list contains a cycle, with both the tail leading into it (n // 2 nodes) and the loop itself (n - n // 2 nodes) growing linearly with n. A traversal that stops on the first repeated node therefore still visits all n nodes first, so the answer is always True and the work is proportional to n rather than to where the cycle happens to sit",
        "generalized_note": "uncapped: any number of nodes, values any ints. The structure is part of the problem, not a cap — the list is still either acyclic or ends in a cycle back to one of its own nodes. Note the stated 1000-node maximum is a cap, so it goes: a solution that walks a fixed number of steps and then declares a cycle is correct only under that cap, and uncapped it reports a cycle on any longer acyclic list. The visited set is keyed on node objects, so its hashes come from CPython's identity hash and the input cannot choose them.",
    },
    "longest-consecutive-sequence": {
        "entry": "longestConsecutive",
        "scalable": True,
        "generate": _gen_longest_consecutive_sequence,
        "adversarial": _adv_longest_consecutive_sequence,
        "adversarial_note": "n distinct multiples of 2**61-1, all hashing to 0, so every set insert/lookup collides",
        "scaling_note": "n = array length; values are a shuffled permutation of range(n), so the value range scales with n too",
        "generalized_note": "uncapped: any array length, values any ints \u2014 including adversarially colliding ones, which the stated -10^9..10^9 range would have made impossible.",
    },
    "longest-substring-without-duplicates": {
        "entry": "lengthOfLongestSubstring",
        "scalable": True,
        "generate": _gen_longest_substring_without_duplicates,
        "adversarial": None,
        "adversarial_note": None,
        "scaling_note": "n = length of the input string; characters are drawn uniformly from the 26 lowercase letters, independent of n. That bounds every duplicate-free window at 26 characters however large n gets (the longest is typically well under that), so a submission whose per-step work is proportional to the current window length still does a bounded amount of work per starting position and comes out linear in n -- the window length is not a second size dimension on this input",
        "generalized_note": "uncapped: any string length. The character set is part of the problem, not a cap -- input stays printable ASCII, letters, digits, symbols and spaces. Note that even that wider alphabet lets duplicate-free windows grow with n, so a solution that rescans its whole window per step is quadratic there while measuring linear on lowercase-only input.",
    },
    "lowest-common-ancestor-in-binary-search-tree": {
        "entry": "lowestCommonAncestor",
        "scalable": True,
        "generate": _gen_lowest_common_ancestor_in_binary_search_tree,
        "build": _build_lowest_common_ancestor_in_binary_search_tree,
        "adversarial": None,
        "adversarial_note": None,
        "scaling_note": "n = number of nodes; the tree is a balanced BST built by taking each subtree's middle value as its root, so its depth is ceil(log2(n)). The n values are sampled without replacement from -10n..10n and sorted, so the value range scales with n and no value repeats. p and q are pinned to the quarter and three-quarter in-order positions, which puts them in opposite subtrees of the root with neither an ancestor of the other, so the answer is always the root itself. That is a deliberate trade: choosing the pair at random makes the cost bimodal for a solution that decides ancestry by comparing whole subtrees, and the ladder then measures the coin flip rather than n. State it plainly in both directions -- against this input a whole-subtree comparison scans a constant fraction of the n nodes and so grows linearly, while a solution that descends from the root comparing p.val and q.val against the current node stops at the first node it looks at and does constant work, not the log n its descent would cost for a deeper answer",
        "generalized_note": "uncapped: any number of nodes, values any ints. The BST ordering is part of the problem, not a cap -- every node's left subtree still holds only smaller values and its right subtree only larger ones, values stay distinct, and p and q are still both present in the tree. The stated node-count maximum is a cap and goes, so the tree may be arbitrarily skewed and its depth linear in n.",
    },
    "lru-cache": {
        "entry": None,
        "scalable": False,
        "generate": _gen_lru_cache,
        "adversarial": None,
        "adversarial_note": None,
        "scaling_note": "not benchmarked: this is a design problem -- the submission defines an LRUCache class with get/put rather than a Solution class with one entry method, so benchmark.py's loader has nothing to drive; the generator builds a capacity of n // 8 and an n-operation sequence (~50% get, ~50% put, keys drawn uniformly from 0..n // 4 so both hits and misses are common and eviction actually fires) purely as documentation of the input shape",
        "generalized_note": "uncapped: any number of operations, any capacity, keys and values any ints. The contract is part of the problem, not a cap — get still returns -1 for a key that is absent or already evicted, put still evicts the least recently used entry once the cache is over capacity, and both count as uses for recency purposes. The stated O(1)-per-operation expectation is part of the problem too.",
    },
    "max-water-container": {
        "entry": "maxArea",
        "scalable": True,
        "generate": _gen_max_water_container,
        "adversarial": None,
        "adversarial_note": None,
        "scaling_note": "n = number of heights; each height is drawn uniformly from the fixed range 1..1000, independent of n, so no height is 0 and the tallest lines are spread through the array rather than sitting at its ends. An all-pairs submission therefore always does the full n(n-1)/2 comparisons, and a two-pointer sweep always does its full n steps -- neither shape gets an early exit from the input",
        "generalized_note": "uncapped: any number of lines, heights any non-negative ints. The geometry is part of the problem, not a cap — the area of a pair is still (j - i) * min(heights[i], heights[j]), and the lines stay at unit spacing in input order.",
    },
    "merge-k-sorted-linked-lists": {
        "entry": "mergeKLists",
        "scalable": True,
        "generate": _gen_merge_k_sorted_linked_lists,
        "build": _build_merge_k_sorted_linked_lists,
        "adversarial": None,
        "adversarial_note": None,
        "scaling_note": "n = total number of nodes, split across k = isqrt(n) chains of near-equal length (never empty), so the chain count and each chain's length both grow like sqrt(n) -- this is the one problem here with a second size dimension, and pinning k to n this way is what makes it visible: a submission that rescans every chain head to pick each output node is O(n * k) = O(n**1.5) on this ladder, where a heap or pairwise merge stays O(n log k). Node values are drawn uniformly from 0..4n and each chain is sorted ascending; that range scales with n so ties between chain heads stay rare, which matters because these submissions short-circuit their head scan on a value equal to the last one emitted. The submissions rewire the input chains in place, so the `build` hook hands each timed repeat freshly built chains",
        "generalized_note": "uncapped: any number of chains, each of any length, node values any ints; chains may be empty and the input list itself may be empty. Sortedness is part of the problem, not a cap — every input chain is ascending and the merged result must be too. Note k is a size dimension of its own: work that is linear in k per output node is O(n * k), which only passes for linear when k is treated as a constant.",
    },
    "merge-two-sorted-linked-lists": {
        "entry": "mergeTwoLists",
        "scalable": True,
        "generate": _gen_merge_two_sorted_linked_lists,
        "build": _build_merge_two_sorted_linked_lists,
        "adversarial": None,
        "adversarial_note": None,
        "scaling_note": "n = total number of nodes across both lists, split n // 2 and n - n // 2; each list's values are drawn independently from the fixed range -1000..1000 and then sorted, so the two interleave throughout instead of one being entirely below the other -- meaning the merge alternates between the chains rather than exhausting one and splicing the rest",
        "generalized_note": "uncapped: both lists any length, node values any ints. Sortedness is part of the problem, not a cap — both inputs are still ascending, and the result must be too.",
    },
    "min-cost-climbing-stairs": {
        "entry": "minCostClimbingStairs",
        "scalable": True,
        "generate": _gen_min_cost_climbing_stairs,
        "adversarial": None,
        "adversarial_note": None,
        "scaling_note": "n = number of steps, floored at 2 because that is where the recurrence starts (the ladder never goes near that floor). Each cost is drawn uniformly from the fixed range 0..1000, independent of n. The costs do not move the measurement -- every submission here is a single right-to-left pass doing one min per step, and both sides of that comparison cost the same -- so only the length matters; what separates the submissions is allocation rather than branching, two of them building an n-element table and three carrying two running values",
        "generalized_note": "uncapped: any number of steps, costs any non-negative ints. The movement rule is part of the problem, not a cap -- you start at step 0 or step 1, a step's cost is paid on leaving it, and each move goes one or two steps forward until you are past the end.",
    },
    "minimum-stack": {
        "entry": None,
        "scalable": False,
        "generate": _gen_minimum_stack,
        "adversarial": None,
        "adversarial_note": None,
        "scaling_note": "not benchmarked: this is a design problem -- the submission defines a MinStack class with push/pop/top/getMin rather than a Solution class with one entry method, so benchmark.py's loader has nothing to drive; the generator builds an n-operation sequence (~60% push, ~40% pop, never popping an empty stack) purely as documentation of the input shape",
        "generalized_note": "uncapped: any number of operations, values any ints. The preconditions are part of the problem \u2014 pop/top/getMin are still only called on a non-empty stack.",
    },
    "missing-number": {
        "entry": "missingNumber",
        "scalable": True,
        "generate": _gen_missing_number,
        "adversarial": None,
        "adversarial_note": None,
        "scaling_note": "n = array length; the array is a shuffled permutation of 0..n-1, so the absent value is always n itself. That is pinned deliberately rather than placed at random: the submission scans with `i not in nums` and stops at the gap, so a randomly placed gap would make each rung of the ladder measure where the gap happened to sit instead of n. As pinned, every i below n is found after about n / 2 comparisons before the final scan for n fails outright, so the work is about n**2 / 2 and the particular shuffle does not change it. The scan is over a list, not a set, so nothing here hashes",
        "generalized_note": "uncapped: any array length. The value range is the problem's structure rather than a size limit -- the array still holds n distinct integers drawn from 0..n, with exactly one of that range absent, which is what makes the sum and xor solutions work at all.",
    },
    "non-cyclical-number": {
        "entry": "isHappy",
        "scalable": True,
        "generate": _gen_non_cyclical_number,
        "adversarial": None,
        "adversarial_note": None,
        "scaling_note": "n = the input integer itself, which is the entire input -- the entry takes one integer, so the size being varied is that integer's value rather than any length, and the generator samples nothing. What grows along the ladder is only the digit count, about log10(n): 3 digits at the first rung and 7 at the last. That also bounds the whole computation. The first digit-square sum drops any input below 81 * digits -- under 600 anywhere on this ladder -- and from there the sequence reaches 1 or the fixed 4-16-37-58-89-145-42-20 cycle within a couple of dozen steps whatever it started from. So the work is one digit pass proportional to log10(n) plus a tail whose cost does not depend on n at all, and the whole ladder runs in the low microseconds. Read the curve as flat: the fitted polynomial and its r^2 are describing timer resolution and a logarithmic term, not a growth rate",
        "generalized_note": "uncapped: any positive integer, of any size. The step is part of the problem, not a cap -- happiness is still defined by repeatedly summing the squares of the decimal digits until reaching 1 or repeating. Note that n is a value rather than a length: writing it down takes about log10(n) digits, so one pass over those digits is linear in the size of the input, while everything after that first pass is bounded by a constant however large n was.",
    },
    "number-of-one-bits": {
        "entry": "hammingWeight",
        "scalable": True,
        "generate": _gen_number_of_one_bits,
        "adversarial": None,
        "adversarial_note": None,
        "scaling_note": "n = the input's width in bits, not an array length: the generator draws a uniformly random n-bit integer with its top bit forced set, so the value really is n bits wide and about half of them are ones. This is the stated 32-bit input uncapped, and it is the only thing here that can scale at all. The two submissions separate sharply on that ladder: the divide-by-2 loop runs one iteration per bit and each `n // 2` rebuilds an O(n)-bit integer, so it is quadratic in n and the per-size cap truncates it partway up, while `int.bit_count()` reads the value a machine word at a time and walks the whole ladder without leaving the microseconds -- about 80 nanoseconds at the bottom and ten microseconds at the top. Only its last few rungs are big enough to outweigh the fixed cost of the call, so its curve reads flatter than the linear scan it actually is",
        "generalized_note": "uncapped: any non-negative integer, of any width. What is counted is part of the problem, not a cap -- still the number of set bits. Note the stated 32-bit width is exactly what makes this constant-time as posed; once the width is the input size, a per-bit loop is no longer free, and the floor is reading the input, which is linear in the number of bits.",
    },
    "plus-one": {
        "entry": "plusOne",
        "scalable": True,
        "generate": _gen_plus_one,
        "adversarial": None,
        "adversarial_note": None,
        "scaling_note": "n = the number of digits; the leading digit is drawn from 1..9 and never 0, so the list really is n digits wide, and the remaining n-1 are uniform over 0..9. That makes this the easy case for the addition itself: a carry stops at the last digit nine times in ten and reaches the front with probability 10**-n, so a digit-by-digit submission does constant expected work and only an all-nines input would make it walk the whole list. What the ladder measures is therefore whatever else a submission does with all n digits -- and the one here joins them into a string, converts that to a single n-digit integer, adds 1 and formats it back, paying big-integer conversion on every call rather than the constant the problem needs. Note the ladder truncates rather than completing: CPython refuses str-to-int conversions past 4300 digits by default, so that submission raises ValueError at the first rung above it (n = 8192) and only the sizes below are reported",
        "generalized_note": "uncapped: any number of digits. The representation is part of the problem, not a cap -- still one decimal digit per element, most significant first, no leading zero, so the array is the number and there is no fixed-width integer type to fall back on. Note the answer can be one digit longer than the input (all nines), and that a carry chain is the only thing that forces more than constant work.",
    },
    "pow-x-n": {
        "entry": "myPow",
        "scalable": True,
        "generate": _gen_pow_x_n,
        "adversarial": None,
        "adversarial_note": None,
        "scaling_note": "n = the exponent, a value rather than a length: the ladder varies the second argument itself, and the only thing that grows with it is its bit length, 9 at the first rung and 21 at the last. Every ladder size is an exact power of two, which is the cheap case for a repeated-squaring submission -- the exponent has one set bit, so it squares bit_length-1 times and recurses once more on an exponent of 0, where an all-ones exponent would make it recurse once per bit instead. The base is drawn with a random sign and a uniform magnitude in 0.5..1.0, always inside the unit interval, so squaring underflows smoothly toward 0.0; a base above 1 raises OverflowError well before the top of the ladder, which is the same reason the stated problem keeps x**n bounded. One caution for reading the numbers: the bit length spans only 9 to 21 across the whole ladder, a factor of 2.3, while n itself spans a factor of 4096, and every submission here finishes in a few microseconds or less at every rung. So the fitted slope and r^2 describe timer resolution and per-call overhead as much as they describe the computation, and anything growing only with the bit length has very little room to separate itself from a flat line",
        "generalized_note": "uncapped: any base and any exponent, positive or negative. The definition is part of the problem, not a cap -- still x raised to the n, with a negative exponent meaning the reciprocal. Note that n is a value rather than a length, so it costs only about log2(|n|) bits to write down: a solution that multiplies n times is already exponential in the size of its input, where repeated squaring is linear in it. Uncapping the result range is also what makes float overflow and underflow real considerations rather than something the stated constraints rule out.",
    },
    "products-of-array-discluding-self": {
        "entry": "productExceptSelf",
        "scalable": True,
        "generate": _gen_products_of_array_discluding_self,
        "adversarial": None,
        "adversarial_note": None,
        "scaling_note": "n = array length; each element is drawn from the fixed set {1, -1}, independent of n",
        "generalized_note": "uncapped: any array length, values any ints. The 32-bit-product guarantee is a cap, so it goes: products may be arbitrarily large, though arithmetic is still counted as unit cost. Division-by-zero handling remains the solution's problem.",
    },
    "remove-node-from-end-of-linked-list": {
        "entry": "removeNthFromEnd",
        "scalable": True,
        "generate": _gen_remove_node_from_end_of_linked_list,
        "build": _build_remove_node_from_end_of_linked_list,
        "adversarial": None,
        "adversarial_note": None,
        "scaling_note": "n = number of nodes; the 1-based position from the end to remove is drawn uniformly from 1..n, so it is not pinned to either end and scales with n. Both shapes of solution therefore traverse a number of nodes proportional to n -- a length count followed by a second walk, or a two-pointer pass whose gap is the position itself",
        "generalized_note": "uncapped: any list length, node values any ints. The precondition is part of the problem, not a cap — the position removed is still between 1 and the list length.",
    },
    "reorder-linked-list": {
        "entry": "reorderList",
        "scalable": True,
        "generate": _gen_reorder_linked_list,
        "build": _build_one_list,
        "adversarial": None,
        "adversarial_note": None,
        "scaling_note": "n = number of nodes; values are drawn from the fixed range -1000..1000, independent of n, and the reordering work depends only on the node count, not the values. The submission rewires in place, so the `build` hook hands it a freshly built, un-reordered chain before each timed repeat",
        "generalized_note": "uncapped: any list length, node values any ints. The interleaving order is part of the problem, not a cap, and the reordering is still done by rewiring nodes rather than by moving values.",
    },
    "reverse-a-linked-list": {
        "entry": "reverseList",
        "scalable": True,
        "generate": _gen_reverse_a_linked_list,
        "build": _build_one_list,
        "adversarial": None,
        "adversarial_note": None,
        "scaling_note": "n = number of nodes; values are drawn from the fixed range -1000..1000, independent of n. Both submissions reverse in place, so the `build` hook hands each timed repeat a freshly built, un-reversed chain -- without that, repeats 2 and 3 would be measuring an already-reversed list, which is O(1) work from the old head",
        "generalized_note": "uncapped: any list length, node values any ints. A solution that assumes a maximum length is correct only under NeetCode's constraints.",
    },
    "reverse-bits": {
        "entry": "reverseBits",
        "scalable": False,
        "generate": _gen_reverse_bits,
        "adversarial": None,
        "adversarial_note": None,
        "scaling_note": "not scalable: the input is always a 32-bit integer and the submission bakes that width in, formatting to exactly 32 binary digits, so there is no size to vary; the generator draws a uniformly random 32-bit value purely as documentation of the input shape",
        "generalized_note": "nothing to uncap: reversal is defined against a fixed width, and 32 bits is that definition rather than a size limit -- the same value reversed under a different width is a different number. Any solution here is constant-work by construction.",
    },
    "reverse-integer": {
        "entry": "reverse",
        "scalable": False,
        "generate": _gen_reverse_integer,
        "adversarial": None,
        "adversarial_note": None,
        "scaling_note": "not scalable: the input is a single integer pinned to the signed 32-bit range, and both submissions bake that in -- one compares the reversed value against 2**31-1 and 2**31, the other rejects anything past ten digits outright -- so there is no size to vary, and a larger input would be answered with 0 rather than worked on; the generator draws a uniformly random value from -2**31..2**31-1 purely as documentation of the input shape",
        "generalized_note": "nothing to uncap in the size: the 32-bit range is the problem rather than a limit on it -- overflowing that range is exactly what the answer is defined to report as 0, so removing the range removes the question. What is left is a digit reversal over about log10(|x|) digits, linear in the size of the input as written.",
    },
    "same-binary-tree": {
        "entry": "isSameTree",
        "scalable": True,
        "generate": _gen_same_binary_tree,
        "build": _build_same_binary_tree,
        "adversarial": None,
        "adversarial_note": None,
        "scaling_note": "n = number of nodes in each of the two trees, so 2n nodes are built per call; both are complete trees (level-order fill) over the same n values, which makes them equal -- the comparison never short-circuits on a mismatch and always visits all n node pairs, this problem's worst case. Depth stays floor(log2(n)) and values are drawn uniformly from the fixed range -1000..1000, independent of n",
        "generalized_note": "uncapped: both trees any size, values any ints, any shape. Equality is part of the problem, not a cap -- the trees must match in structure and in values. Note that unequal trees let a solution stop at the first difference, so the equal case measured here is the worst case, and a skewed tree makes recursion depth linear in n.",
    },
    "search-2d-matrix": {
        "entry": "searchMatrix",
        "scalable": True,
        "generate": _gen_search_2d_matrix,
        "adversarial": None,
        "adversarial_note": None,
        "scaling_note": "n = total number of cells; the matrix is isqrt(n) rows by n // isqrt(n) columns, so both dimensions grow like sqrt(n) and neither is a constant a solution can be linear in for free. Cells are sampled without replacement from -10n..10n and laid out ascending in row-major order, so the whole matrix reads as one sorted array (each row ascending, and every value larger than the last value of the row above), and the target is always one of the cells -- never absent, so every run finds it. Expect a flat, near-zero slope from a binary search: O(log n) grows by a constant per doubling of n rather than scaling with it, while a full scan of every cell would come out linear. Note the whole ladder runs in single-digit microseconds for a binary search, close enough to the clock's resolution that the fit is noisy even when the shape is unmistakably logarithmic -- read the flatness, not the r^2",
        "generalized_note": "uncapped: any number of rows and columns, values any ints. The ordering is part of the problem, not a cap -- each row is still ascending and each row's first value still exceeds the previous row's last. The target may be absent, in which case the answer is false.",
    },
    "single-number": {
        "entry": "singleNumber",
        "scalable": True,
        "generate": _gen_single_number,
        "adversarial": None,
        "adversarial_note": None,
        "scaling_note": "n = requested array length, rounded down to the nearest odd number: the array is (n - 1) // 2 pairs plus one unpaired value, so an even n yields n - 1 elements, and since every ladder size is a power of two each rung is in fact one element short of its label. Values are sampled without replacement from -4n..4n and then shuffled, so no pair shares a value with another and the pairs are scattered through the array rather than adjacent. Neither of those affects the work -- the submission is one xor per element regardless of order or value -- so only the length moves the measurement",
        "generalized_note": "uncapped: any odd array length, values any ints. The multiplicity contract is part of the problem, not a cap -- every value appears exactly twice except one, which appears once and is the answer.",
    },
    "string-encode-and-decode": {
        "entry": "encode",
        "scalable": True,
        "generate": _gen_string_encode_and_decode,
        "adversarial": None,
        "adversarial_note": None,
        "scaling_note": "n = number of strings; each string's length is drawn from a fixed small range (3-12 chars) independent of n, so total character volume scales linearly with n",
        "generalized_note": "uncapped: any number of strings, each of any length. The character set is part of the problem, not a cap \u2014 input is still drawn from the 256 ASCII characters, so a delimiter outside that set remains safe. Widening to arbitrary Unicode is the statement's separate follow-up, not this generalization.",
    },
    "subsets": {
        "entry": "subsets",
        "scalable": True,
        "sizes": EXPONENTIAL_SIZES,
        "models": EXPONENTIAL_MODELS,
        "generate": _gen_subsets,
        "adversarial": None,
        "adversarial_note": None,
        "scaling_note": "n = array length, and this problem runs a ladder of its own: n steps by 1 from 4 to 21 rather than doubling from 256, because the output is the power set and one doubling of n squares the work -- the default ladder's first rung would ask for 2**256 subsets. Values are sampled without replacement from -2n..2n, since the problem guarantees them distinct; their magnitudes are never read. Two things follow for reading the numbers. The reported log-log slope is meaningless here -- it fits log(time) against log(n), and an exponential curve is not a line in those coordinates -- so read best_fit, which is chosen from 2^n and n * 2^n alongside the usual polynomials. And no solution can do better than exponential: the output alone is 2**n subsets holding n * 2**(n-1) values",
        "generalized_note": "uncapped: any array length, values any ints. Distinctness and the output contract are part of the problem, not caps -- the input has no duplicates, and every subset must appear exactly once, in any order. The stated length maximum is a cap and goes, but note that the output alone is 2**k subsets holding k * 2**(k-1) values in total, so exponential time in the array length is a floor no solution gets under.",
    },
    "subtree-of-a-binary-tree": {
        "entry": "isSubtree",
        "scalable": True,
        "generate": _gen_subtree_of_a_binary_tree,
        "build": _build_subtree_of_a_binary_tree,
        "adversarial": None,
        "adversarial_note": None,
        "scaling_note": "n = number of nodes in the main tree; it is a complete tree (level-order fill) over a shuffled permutation of range(n), so its depth is floor(log2(n)) and every value is distinct. subRoot is a fresh copy of the subtree hanging at the one node a pre-order search reaches last, so the answer is always True but only after every node has been visited -- pinned there rather than chosen at random, since a random target would make each rung of the ladder measure where the match happened to sit instead of n. Because the values are distinct, the per-node equality check fails on its first value comparison everywhere except at the true match (which is a single leaf), so what is measured is the traversal itself and not repeated deep comparisons",
        "generalized_note": "uncapped: both trees any size, values any ints, any shape, and values may repeat. The matching condition is part of the problem, not a cap -- a match is still a node of the main tree whose entire subtree equals subRoot in both structure and values. Note that repeated values are what make a naive scan quadratic: with duplicates the per-node equality check can run deep at every node instead of failing on its first comparison.",
    },
    "sum-of-two-integers": {
        "entry": "getSum",
        "scalable": False,
        "generate": _gen_sum_of_two_integers,
        "adversarial": None,
        "adversarial_note": None,
        "scaling_note": "not scalable: the input is two integers, and the submissions that do the addition by hand bake in a 32-bit width -- formatting to exactly 32 binary digits, or looping over exactly 32 bit positions -- so there is no size to vary; the generator draws a uniformly random pair from the stated -1000..1000 purely as documentation of the input shape",
        "generalized_note": "nothing to uncap in the size: two fixed-width integers is the problem's structure. The constraint that matters here is the one a size cap hides -- the point is to add without using + or -, which the bit-by-bit submissions honour and the ones calling + or sum() do not. Any solution here is constant-work by construction.",
    },
    "three-integer-sum": {
        "entry": "threeSum",
        "scalable": True,
        "generate": _gen_three_integer_sum,
        "adversarial": None,
        "adversarial_note": None,
        "scaling_note": "n = array length; values are sampled without replacement from -n**2..n**2, so the value range grows quadratically with n. That has two deliberate effects: the values are distinct, so no triple is ever produced twice and a submission that dedups by scanning its result list effectively never pays for it; and the number of zero-sum triples grows only linearly with n (25 of them at n = 512), so the answer stays small and the measured time is the search itself -- the O(n log n) sort plus the O(n**2) two-pointer sweep -- rather than the cost of assembling and deduplicating output",
        "generalized_note": "uncapped: any array length, values any ints, and values may repeat. The output contract is part of the problem, not a cap — still every distinct triple summing to zero, each reported once, in any order. Note the number of such triples can itself be quadratic in n, so a solution that dedups by scanning the result list is quadratic in the output size on top of its search cost, and no solution can beat the output size itself.",
    },
    "top-k-elements-in-list": {
        "entry": "topKFrequent",
        "scalable": True,
        "generate": _gen_top_k_elements_in_list,
        "adversarial": _adv_top_k_elements_in_list,
        "adversarial_note": "n distinct multiples of 2**61-1, all hashing to 0, so every dict insert/lookup collides",
        "scaling_note": "n = array length; k (the count requested) is capped at min(5, n), so k stays bounded by a constant and does not grow with n",
        "generalized_note": "uncapped: any array length, values any ints \u2014 including adversarially colliding ones, which the stated -1000..1000 range would have made impossible. k stays between 1 and the number of distinct elements.",
    },
    "trapping-rain-water": {
        "entry": "trap",
        "scalable": True,
        "generate": _gen_trapping_rain_water,
        "adversarial": None,
        "adversarial_note": None,
        "scaling_note": "n = number of bars; each height is drawn uniformly from the fixed range 0..1000, independent of n, so heights repeat freely, zero-height bars do occur (about 1 in 1001) and the tallest bars are spread through the array rather than sitting at its ends. Because that range is fixed, the number of distinct water levels a sweep can encounter is bounded by a constant rather than by n: a submission that refills a level array each time the smaller of its two end heights reaches a new maximum does at most about 1000 refills of O(n) each, so it measures as linear in n -- a bound this input's fixed height range supplies, not the algorithm. Nothing in the input shortens a prefix/suffix scan, either: the maxima are spread through the array, so recomputing max(height[:i]) and max(height[i+1:]) at an index copies and scans two slices whose lengths sum to n, at every index, with no early exit available",
        "generalized_note": "uncapped: any number of bars, heights any non-negative ints. The geometry is part of the problem, not a cap -- the water above index i is still min(tallest bar at or left of i, tallest bar at or right of i) - height[i], floored at 0, and bars stay one unit wide in input order. Note the stated height cap is what bounds a level-sweeping solution's refill count; without it that count scales with the number of distinct heights, which can grow with n.",
    },
    "two-integer-sum": {
        "entry": "twoSum",
        "scalable": True,
        "generate": _gen_two_integer_sum,
        "adversarial": _adv_two_integer_sum,
        "adversarial_note": "n distinct multiples of 2**61-1, all hashing to 0, so every dict insert/lookup collides",
        "scaling_note": "n = array length; values are sampled from a range that scales with n (-10n..10n)",
        "generalized_note": "uncapped: any array length, values and target any ints \u2014 including adversarially colliding ones. Exactly one valid answer still exists.",
    },
    "two-integer-sum-ii": {
        "entry": "twoSum",
        "scalable": True,
        "generate": _gen_two_integer_sum_ii,
        "adversarial": _adv_two_integer_sum_ii,
        "adversarial_note": "n distinct multiples of 2**61-1 (already ascending), all hashing to 0, so every dict insert/lookup collides",
        "scaling_note": "n = array length (input is pre-sorted); values are sampled from a range that scales with n (-10n..10n)",
        "generalized_note": "uncapped: any array length, values and target any ints \u2014 including adversarially colliding ones. The input is still sorted ascending and exactly one valid answer exists.",
    },
    "validate-parentheses": {
        "entry": "isValid",
        "scalable": True,
        "generate": _gen_validate_parentheses,
        "adversarial": None,
        "adversarial_note": None,
        "scaling_note": "n = string length; a uniformly random balanced bracket sequence over ()[]{}",
        "generalized_note": "uncapped: any string length. The bracket alphabet is part of the problem, not a cap.",
    },
    "valid-sudoku": {
        "entry": "isValidSudoku",
        "scalable": False,
        "generate": _gen_valid_sudoku,
        "adversarial": None,
        "adversarial_note": None,
        "scaling_note": "not scalable: input is always a fixed 9x9 board",
        "generalized_note": "nothing to uncap: the 9x9 board is the problem's structure, not a size limit, so the generalized problem is the stated one. Any solution here is constant-work by construction.",
    },
}
