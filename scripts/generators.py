"""Deterministic input generators for benchmarking NeetCode submissions.

Each problem in `PROBLEMS` maps to:
- `entry`: the Solution method to call, or None for a design problem
  whose submission defines no Solution class at all (minimum-stack).
- `scalable`: False for problems benchmark.py can't time -- fixed-shape
  inputs (valid-sudoku is a fixed 9x9 board) and shapes its loader can't
  construct (minimum-stack and every linked-list problem; see their
  scaling_note); benchmark.py skips scaling curves for these, so their
  `generate` documents the input shape rather than feeding a timing run.
- `generate(n, rng)`: returns a positional-args tuple (excluding `self`)
  for `entry`, sized around n. Uses only the passed-in random.Random, so
  it's deterministic for a given seed.
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
input strings (is-anagram keys on single characters — a 26-symbol
alphabet, too small to stress; anagram-groups keys on a computed
26-int signature tuple, not the input strings; string-encode-and-decode
doesn't hash at all; the linked-list problems that hash at all key on
node objects, whose hashes come from CPython's identity hash, so the
input can't choose them). So there's currently no PYTHONHASHSEED=0-dependent
string-collision generator in use. benchmark.py still forces
PYTHONHASHSEED=0 for the whole run as a forward-looking default, since
str/bytes hashing is otherwise randomized per-process and any future
string-keyed adversarial generator would need it fixed before the
interpreter starts.
"""

import random
import string

MERSENNE61 = 2**61 - 1
RPN_VALUE_LIMIT = 10**6


def _int_collisions(n):
    return [k * MERSENNE61 for k in range(n)]


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


def _gen_anagram_groups(n, rng):
    return ([_random_word(rng, 3, 8) for _ in range(n)],)


def _adv_anagram_groups(n):
    base = "abcdefghij"
    words = [base[i % len(base):] + base[:i % len(base)] for i in range(n)]
    return (words,)


def _gen_binary_search(n, rng):
    nums = sorted(rng.sample(range(-n * 10, n * 10 + 1), n))
    return (nums, nums[rng.randrange(n)])


def _gen_buy_and_sell_crypto(n, rng):
    return ([rng.randint(1, 1000) for _ in range(n)],)


def _gen_car_fleet(n, rng):
    target = 2 * n + 1
    position = rng.sample(range(target), n)
    speed = [rng.randint(1, 100) for _ in range(n)]
    return (target, position, speed)


def _gen_copy_linked_list_with_random_pointer(n, rng):
    values = [rng.randint(-1000, 1000) for _ in range(n)]
    randoms = [rng.randrange(n) if rng.random() < 0.75 else None for _ in range(n)]
    return (values, randoms)


def _gen_daily_temperatures(n, rng):
    return ([rng.randint(30, 100) for _ in range(n)],)


def _gen_duplicate_integer(n, rng):
    return (rng.sample(range(n * 4 + 1), n),)


def _adv_duplicate_integer(n):
    return (_int_collisions(n),)


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


def _gen_largest_rectangle_in_histogram(n, rng):
    return ([rng.randint(0, 10000) for _ in range(n)],)


def _gen_linked_list_cycle_detection(n, rng):
    values = [rng.randint(-1000, 1000) for _ in range(n)]
    return (values, n // 2)


def _gen_longest_consecutive_sequence(n, rng):
    nums = list(range(n))
    rng.shuffle(nums)
    return (nums,)


def _adv_longest_consecutive_sequence(n):
    return (_int_collisions(n),)


def _gen_merge_two_sorted_linked_lists(n, rng):
    half = n // 2
    list1 = sorted(rng.randint(-1000, 1000) for _ in range(half))
    list2 = sorted(rng.randint(-1000, 1000) for _ in range(n - half))
    return (list1, list2)


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


def _gen_products_of_array_discluding_self(n, rng):
    return ([rng.choice((1, -1)) for _ in range(n)],)


def _gen_remove_node_from_end_of_linked_list(n, rng):
    values = [rng.randint(-1000, 1000) for _ in range(n)]
    return (values, rng.randint(1, n) if n else 1)


def _gen_reorder_linked_list(n, rng):
    return ([rng.randint(-1000, 1000) for _ in range(n)],)


def _gen_reverse_a_linked_list(n, rng):
    return ([rng.randint(-1000, 1000) for _ in range(n)],)


def _gen_string_encode_and_decode(n, rng):
    return ([_random_word(rng, 3, 12) for _ in range(n)],)


def _gen_top_k_elements_in_list(n, rng):
    k = min(5, n) if n else 0
    return ([rng.randint(0, max(1, n // 4)) for _ in range(n)], k)


def _adv_top_k_elements_in_list(n):
    k = min(5, n) if n else 0
    return (_int_collisions(n), k)


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
        "scalable": False,
        "generate": _gen_add_two_numbers,
        "adversarial": None,
        "adversarial_note": None,
        "scaling_note": "not benchmarked: n = number of digits in each addend, and the generator returns the two digit lists (least-significant first, leading digit never 0) the linked lists would be built from, but the entry takes two ListNode heads. Building real nodes here wouldn't be enough either: every submission allocates result nodes with `ListNode(...)`, a name benchmark.py's loader never defines -- its exec namespace holds only List and Optional -- so the call raises NameError before any timing",
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
    "binary-search": {
        "entry": "search",
        "scalable": True,
        "generate": _gen_binary_search,
        "adversarial": None,
        "adversarial_note": None,
        "scaling_note": "n = array length; values are sampled without replacement from a range that scales with n (-10n..10n) and then sorted ascending, and the target is always one of the n values (never absent), so every run finds its target",
        "generalized_note": "uncapped: any array length, values any ints. Sortedness and uniqueness are part of the problem, not caps, and the target stays one of the values.",
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
    "copy-linked-list-with-random-pointer": {
        "entry": "copyRandomList",
        "scalable": False,
        "generate": _gen_copy_linked_list_with_random_pointer,
        "adversarial": None,
        "adversarial_note": None,
        "scaling_note": "not benchmarked: n = number of nodes, and the generator returns the n values plus, for each node, the index its random pointer targets (None ~25% of the time, for a null random), but the entry takes a Node head. Building real nodes here wouldn't be enough either: the submission allocates each copy with `Node(...)`, a name benchmark.py's loader never defines -- its exec namespace holds only List and Optional -- so the call raises NameError before any timing",
        "generalized_note": "uncapped: any number of nodes, values any ints. The pointer structure is part of the problem, not a cap — each random pointer still targets some node in the list or null. The dict here is keyed on node objects, so its hashes come from CPython's identity hash and the input cannot choose them.",
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
    "duplicate-integer": {
        "entry": "hasDuplicate",
        "scalable": True,
        "generate": _gen_duplicate_integer,
        "adversarial": _adv_duplicate_integer,
        "adversarial_note": "n distinct multiples of 2**61-1, all hashing to 0, so every dict insert/lookup collides",
        "scaling_note": "n = array length; values are sampled without replacement from range(4n+1), so the value range scales with n too",
        "generalized_note": "uncapped: any array length, values any ints \u2014 including values an adversary picks to collide in CPython's hash (multiples of 2**61-1), which the stated -10^9..10^9 range would have made impossible.",
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
    "largest-rectangle-in-histogram": {
        "entry": "largestRectangleArea",
        "scalable": True,
        "generate": _gen_largest_rectangle_in_histogram,
        "adversarial": None,
        "adversarial_note": None,
        "scaling_note": "n = number of bars; each height is drawn uniformly from 0..10000, NeetCode's full stated range for heights[i], independent of n -- so a submission that sweeps every height value from 0 to max(heights) does a bounded (<=10001) number of linear passes: still O(n) in n, but with a constant large enough that the size ladder hits its per-size time cap around n = 2**13 to 2**14, depending on machine speed",
        "generalized_note": "uncapped: any number of bars, heights any non-negative ints. Note the stated 0..10000 height cap is what makes a height-indexed sweep O(n); without it such a solution scales with the largest height instead.",
    },
    "linked-list-cycle-detection": {
        "entry": "hasCycle",
        "scalable": False,
        "generate": _gen_linked_list_cycle_detection,
        "adversarial": None,
        "adversarial_note": None,
        "scaling_note": "not benchmarked: n = number of nodes, and the generator returns the n values plus the index the tail links back to (always n // 2, so every generated list does contain a cycle), but the entry takes a ListNode head. Unlike the other linked-list problems neither submission allocates or rewires nodes, so the blocker here is only that generate cannot hand over the nodes themselves: freshly built node objects compare by identity, and the test suite asserts that two same-seed generate calls return equal args",
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
    "merge-two-sorted-linked-lists": {
        "entry": "mergeTwoLists",
        "scalable": False,
        "generate": _gen_merge_two_sorted_linked_lists,
        "adversarial": None,
        "adversarial_note": None,
        "scaling_note": "not benchmarked: n = total number of nodes, and the generator returns the two ascending value lists (n // 2 and n - n // 2 values, each drawn independently from the fixed range -1000..1000 and then sorted, so the two genuinely interleave) the linked lists would be built from, but the entry takes two ListNode heads. Building real nodes here wouldn't be enough either: the submission splices the two chains together in place, and benchmark.py's _copy_args only deep-copies lists, so best-of-3 would re-time an already-merged chain on repeats 2 and 3",
        "generalized_note": "uncapped: both lists any length, node values any ints. Sortedness is part of the problem, not a cap — both inputs are still ascending, and the result must be too.",
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
        "scalable": False,
        "generate": _gen_remove_node_from_end_of_linked_list,
        "adversarial": None,
        "adversarial_note": None,
        "scaling_note": "not benchmarked: n = number of nodes, and the generator returns the n values the list would be built from plus the 1-based position from the end to remove (drawn uniformly from 1..n, so the removal point is not pinned to either end), but the entry takes a ListNode head. Building real nodes here wouldn't be enough either: the submission allocates a sentinel with `ListNode(next=head)`, a name benchmark.py's loader never defines -- its exec namespace holds only List and Optional -- so the call raises NameError before any timing",
        "generalized_note": "uncapped: any list length, node values any ints. The precondition is part of the problem, not a cap — the position removed is still between 1 and the list length.",
    },
    "reorder-linked-list": {
        "entry": "reorderList",
        "scalable": False,
        "generate": _gen_reorder_linked_list,
        "adversarial": None,
        "adversarial_note": None,
        "scaling_note": "not benchmarked: n = number of nodes, and the generator returns the n values the list would be built from, but the entry takes a ListNode head. Building real nodes here wouldn't be enough either: the submission rewires the list in place and returns None, and benchmark.py's _copy_args only deep-copies lists, so best-of-3 would re-time an already-reordered chain on repeats 2 and 3",
        "generalized_note": "uncapped: any list length, node values any ints. The interleaving order is part of the problem, not a cap, and the reordering is still done by rewiring nodes rather than by moving values.",
    },
    "reverse-a-linked-list": {
        "entry": "reverseList",
        "scalable": False,
        "generate": _gen_reverse_a_linked_list,
        "adversarial": None,
        "adversarial_note": None,
        "scaling_note": "not benchmarked: n = number of nodes, and the generator returns the n values the list would be built from, but the entry takes a linked-list head -- passing the list straight through raises AttributeError on `.next`. Building real nodes here wouldn't be enough either: both submissions reverse in place, and benchmark.py's _copy_args only deep-copies lists, so best-of-3 would re-time an already-reversed chain (O(1) from the old head) on repeats 2 and 3",
        "generalized_note": "uncapped: any list length, node values any ints. A solution that assumes a maximum length is correct only under NeetCode's constraints.",
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
    "top-k-elements-in-list": {
        "entry": "topKFrequent",
        "scalable": True,
        "generate": _gen_top_k_elements_in_list,
        "adversarial": _adv_top_k_elements_in_list,
        "adversarial_note": "n distinct multiples of 2**61-1, all hashing to 0, so every dict insert/lookup collides",
        "scaling_note": "n = array length; k (the count requested) is capped at min(5, n), so k stays bounded by a constant and does not grow with n",
        "generalized_note": "uncapped: any array length, values any ints \u2014 including adversarially colliding ones, which the stated -1000..1000 range would have made impossible. k stays between 1 and the number of distinct elements.",
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
