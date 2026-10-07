# Written by Claude Opus 5.5 (Anthropic), not by April: a port of the
# algorithm Claude discovered and Alman & Vassilevska Williams wrote up,
# made to see what the analysis pipeline says about it.

from typing import List
from bisect import bisect_left
from itertools import combinations
from math import isqrt
from operator import add, sub, neg, mul

# 3SUM via Alman & Vassilevska Williams, "Truly Subquadratic 3SUM and Truly
# Subcubic APSP via Triangles in Sparse Lopsided Graphs" (arXiv 2610.06783v1):
#   3SUM -> Convolution-3SUM -> Exact Triangle   (Theorem 21(a), [CH20, VW13])
#   Exact Triangle -> Lop-AE-SparseTri           (Theorem 17)
#   Lop-AE-SparseTri -> wanted entries of XY     (Corollary 15 -> Theorem 5)
#
# SIZE CUTOFF: with at most THEOREM5_MAX_DISTINCT distinct values, every
# Lop-AE-SparseTri instance is answered by Theorem 5 (the faithful path).
# Above it, the same instances are answered by the straightforward O(|W| D)
# common-neighbour test of Section 3.1 (Remark 20 allows any oracle); every
# other stage still runs.  Theorem 5's saving needs L = 19m and N >= D^18 =
# 2^72 (Section 2.4.4); at the toy parameters below it spends 31-100 leaf
# products per wanted entry, against D = 16 for an inner product, which pure
# Python cannot afford at n = 1000.
THEOREM5_MAX_DISTINCT = 200

try:
    _popcount = int.bit_count
except AttributeError:
    def _popcount(x):
        return bin(x).count("1")


class ThinProduct:
    # Theorem 5 with m = 2, so D = 4^m = 16 (D = 4 is too small for Theorem 17,
    # which needs D >= 16 to have a prime in [sqrt(D)/2, sqrt(D))).  L = 4
    # levels instead of the paper's L = 19m = 38 (10^38 leaves per tile):
    # N0 = 3^(L-m) = 9, K = C(4,2) = 6, K0 = 2, so one run of Full (a tile)
    # covers 18 x 18 entries.  Lemmata 7-10 hold for every L >= m; L = 19m and
    # N >= D^18 only drive the running-time bound (Lemma 11, Section 2.4.4).
    # Any K0^2 of the subsets may be fixed (Section 2.3.4): we take the four
    # whose inner levels lie deepest, so Pruned's last levels see whole rows of
    # leaves rather than single ones.
    m = 2
    L = 4

    def __init__(self):
        L, m = self.L, self.m
        self.D = 4 ** m
        self.N0 = 3 ** (L - m)
        self.subsets = sorted(combinations(range(L), m), key=lambda Q: Q[::-1], reverse=True)
        self.K0 = isqrt(len(self.subsets))
        self.band = self.K0 * self.N0
        self.trace = None
        self._identity()
        self._strings()

    def _identity(self):
        # Lemma 6.  Left variables x1, x2, x3, p11, p12, p21, p22 -> 0..6 (right
        # y's and q's likewise), terms P_ij -> 3(i-1) + (j-1) and P0 -> 9,
        # outputs z_ij and z0 numbered like the terms.
        inner = {(0, 0): 3, (0, 1): 4, (1, 0): 5, (1, 1): 6}
        phat = [[{} for _ in range(3)] for _ in range(3)]
        qhat = [[{} for _ in range(3)] for _ in range(3)]
        for (i, j), v in inner.items():
            phat[i][j][v] = 1
            phat[2][j][v] = -1
            qhat[i][j][v] = 1
            qhat[i][2][v] = -1
        phi, psi = [], []
        for i in range(3):
            for j in range(3):
                phi.append(dict(phat[i][j]))
                phi[-1][i] = 1
                psi.append(dict(qhat[i][j]))
                psi[-1][j] = 1
        phi.append({0: -1, 1: -1, 2: -1})
        psi.append({0: 1, 1: 1, 2: 1})
        self.phi = [sorted(f.items()) for f in phi]
        self.psi = [sorted(f.items()) for f in psi]

    def _strings(self):
        # Sections 2.3.3-2.3.4: grid cell (r, c) of a tile gets subset q = r K0 + c;
        # left strings with that inner set index row block r of the band of X,
        # right strings column block c of Y, output strings the block product.
        L, K0, N0 = self.L, self.K0, self.N0
        used = {Q: q for q, Q in enumerate(self.subsets[:K0 * K0])}
        self.left_slots, self.right_slots = [], []
        for idx in range(7 ** L):
            u = [idx // 7 ** (L - 1 - l) % 7 for l in range(L)]
            q = used.get(tuple(l for l in range(L) if u[l] >= 3))
            if q is None:
                continue
            outer = inner = 0
            for v in u:
                if v >= 3:
                    inner = 4 * inner + v - 3
                else:
                    outer = 3 * outer + v
            self.left_slots.append((idx, q // K0 * N0 + outer, inner))
            self.right_slots.append((idx, inner, q % K0 * N0 + outer))
        self.out_code = {}
        for q, Q in enumerate(self.subsets[:K0 * K0]):
            outer_levels = [l for l in range(L) if l not in Q]
            for row in range(N0):
                for col in range(N0):
                    w = [9] * L
                    for pos, l in enumerate(outer_levels):
                        sh = 3 ** (len(outer_levels) - 1 - pos)
                        w[l] = 3 * (row // sh % 3) + col // sh % 3
                    code = 0
                    for z in w:
                        code = 10 * code + z
                    self.out_code[q // K0 * N0 + row, q % K0 * N0 + col] = code

    def encode(self, arr, forms):
        # Section 2.4.1: the 10^L leaf values Phi_tau(a) (or Psi_tau(b)) by Yates'
        # algorithm, one level per pass, indexed tau_1 ... tau_L in base 10.
        cur = arr
        for _ in range(self.L):
            V = [cur[s::7] for s in range(7)]
            out = []
            for form in forms:
                (s0, c0), rest = form[0], form[1:]
                acc = V[s0] if c0 == 1 else map(neg, V[s0])
                for s, c in rest:
                    acc = map(add if c == 1 else sub, acc, V[s])
                out.extend(acc)
            cur = out
        return cur

    def encode_rows(self, X, band):
        # Rows past the end of X are the zero padding of Section 2.3.4.
        a = [0] * 7 ** self.L
        row0, n = band * self.band, len(X)
        for idx, r, c in self.left_slots:
            if row0 + r < n:
                a[idx] = X[row0 + r][c]
        return self.encode(a, self.phi)

    def encode_cols(self, Y, band):
        b = [0] * 7 ** self.L
        col0, n = band * self.band, len(Y[0])
        for idx, r, c in self.right_slots:
            if col0 + c < n:
                b[idx] = Y[r][col0 + c]
        return self.encode(b, self.psi)

    def pruned(self, EA, EB, tau, k, S):
        # Section 2.4.2: Pruned_{tau}(S), S a sorted list of output strings of
        # length L - k (base 10), returning the outputs of vertex tau on S.
        # S_{P_ij} = S_{z_ij} u S_{z0}, S_{P0} = S_{z0}; c_{z0} = sum of all C.
        if k == self.L - 2:
            return self._bottom(EA, EB, tau, S)
        div = 10 ** (self.L - k - 1)
        bounds = [bisect_left(S, d * div) for d in range(10)] + [len(S)]
        out = [0] * len(S)
        lo9 = bounds[9]
        z0 = [w - 9 * div for w in S[lo9:]]
        child = 10 * tau
        if not z0:
            for lam in range(9):
                lo, hi = bounds[lam], bounds[lam + 1]
                if lo < hi:
                    off = lam * div
                    out[lo:hi] = self.pruned(EA, EB, child + lam, k + 1, [w - off for w in S[lo:hi]])
            return out
        acc = self.pruned(EA, EB, child + 9, k + 1, z0)
        for lam in range(9):
            lo, hi = bounds[lam], bounds[lam + 1]
            if lo == hi:
                acc = list(map(add, acc, self.pruned(EA, EB, child + lam, k + 1, z0)))
                continue
            off = lam * div
            g = [w - off for w in S[lo:hi]]
            Sl = sorted(set(g).union(z0))
            val = dict(zip(Sl, self.pruned(EA, EB, child + lam, k + 1, Sl)))
            out[lo:hi] = [val[w] for w in g]
            acc = [x + val[w] for x, w in zip(acc, z0)]
        out[lo9:] = acc
        return out

    def _bottom(self, EA, EB, tau, S):
        # Pruned at depth L - 2 with its children unrolled.  Child lam gets
        # S_{z_lam} u S_{z0} and needs all ten of its leaves iff that holds z0,
        # else one leaf per string.  So in the 10 x 10 block of leaves below,
        # (z_r, z0) is row r, (z0, z_c) column c and (z0, z0) everything.
        b = 100 * tau
        trace = self.trace
        if S[-1] == 99:
            prods = list(map(mul, EA[b:b + 100], EB[b:b + 100]))
            if trace is not None:
                trace.extend(range(b, b + 100))
            out = []
            for w in S:
                r, c = divmod(w, 10)
                if c == 9:
                    out.append(sum(prods[10 * r:10 * r + 10]) if r < 9 else sum(prods))
                else:
                    out.append(sum(prods[c::10]) if r == 9 else prods[w])
            return out
        lo9 = bisect_left(S, 90)
        if any(w % 10 != 9 for w in S[:lo9]):
            return self._bottom_by_child(EA, EB, tau, S, lo9)
        rows = {}
        out = []
        for w in S[:lo9]:
            base = b + w - 9
            rows[w // 10] = vals = list(map(mul, EA[base:base + 10], EB[base:base + 10]))
            out.append(sum(vals))
            if trace is not None:
                trace.extend(range(base, base + 10))
        for w in S[lo9:]:
            c = b + w - 90
            if not rows:
                out.append(sum(map(mul, EA[c:c + 100:10], EB[c:c + 100:10])))
                if trace is not None:
                    trace.extend(range(c, c + 100, 10))
                continue
            s = 0
            for lam in range(10):
                vals = rows.get(lam)
                if vals is None:
                    s += EA[c + 10 * lam] * EB[c + 10 * lam]
                    if trace is not None:
                        trace.append(c + 10 * lam)
                else:
                    s += vals[w - 90]
            out.append(s)
        return out

    def _bottom_by_child(self, EA, EB, tau, S, lo9):
        # The same, child by child, for sets with a string (z_r, z_c).
        bounds = [bisect_left(S, 10 * d, 0, lo9) for d in range(9)] + [lo9, lo9]
        z0 = [w - 90 for w in S[lo9:]]
        out = [0] * len(S)
        acc = [0] * len(z0)
        trace = self.trace
        b = 100 * tau
        for lam in range(10):
            lo, hi = bounds[lam], bounds[lam + 1]
            if lo == hi and not z0:
                b += 10
                continue
            g = [w - 10 * lam for w in S[lo:hi]]
            if g and g[-1] == 9:
                vals = list(map(mul, EA[b:b + 10], EB[b:b + 10]))
                vals[9] = sum(vals)
                if trace is not None:
                    trace.extend(range(b, b + 10))
            else:
                vals = [0] * 9
                for z in set(g).union(z0):
                    vals[z] = EA[b + z] * EB[b + z]
                    if trace is not None:
                        trace.append(b + z)
            if g:
                out[lo:hi] = [vals[z] for z in g]
            if z0:
                acc = [x + vals[z] for x, z in zip(acc, z0)]
            b += 10
        out[lo9:] = acc
        return out

    def wanted_entries(self, X, Y, W, x_enc=None, y_enc=None):
        # Theorem 5 / Section 2.4.4: (XY)[I, J] for (I, J) in W, X an N x D and Y
        # a D x N' matrix (lists of rows).  x_enc / y_enc cache band encodings.
        band, out_code = self.band, self.out_code
        tiles = {}
        for I, J in W:
            R, i = divmod(I, band)
            C, j = divmod(J, band)
            tiles.setdefault((R, C), []).append((out_code[i, j], I, J))
        x_enc = {} if x_enc is None else x_enc
        y_enc = {} if y_enc is None else y_enc
        res = {}
        for (R, C), lst in tiles.items():
            EA = x_enc.get(R)
            if EA is None:
                EA = x_enc[R] = self.encode_rows(X, R)
            EB = y_enc.get(C)
            if EB is None:
                EB = y_enc[C] = self.encode_cols(Y, C)
            lst.sort()
            for (_, I, J), v in zip(lst, self.pruned(EA, EB, 0, 0, [w for w, _, _ in lst])):
                res[I, J] = v
        return res


class Theorem5Oracle:
    # Corollary 15: Lop-AE-SparseTri on the biadjacency matrices (middle part
    # padded to D columns); a query pair lies in a triangle iff its entry of XY,
    # its number of common neighbours, is nonzero.
    def __init__(self):
        self.tp = ThinProduct()
        self.cache = {}

    def _side(self, key, build):
        hit = self.cache.get(key) if key is not None else None
        if hit is None:
            hit = (build(), {})
            if key is not None:
                self.cache[key] = hit
        return hit

    def __call__(self, Xmask, Ymask, chunk, x_key=None, y_key=None):
        D = self.tp.D
        X, xe = self._side(x_key, lambda: [[m >> d & 1 for d in range(D)] for m in Xmask])
        Y, ye = self._side(y_key, lambda: [[m >> d & 1 for m in Ymask] for d in range(D)])
        n = max(len(Xmask), len(Ymask))
        lim = max(1, n * n // isqrt(D))
        acc = []
        for i in range(0, len(chunk), lim):
            part = chunk[i:i + lim]
            vals = self.tp.wanted_entries(X, Y, [(a, b) for a, b, _ in part], xe, ye)
            acc.extend(e for e in part if vals[e[0], e[1]])
        return acc


def straightforward_oracle(Xmask, Ymask, chunk, x_key=None, y_key=None):
    # Section 3.1's brute force for Lop-AE-SparseTri, O(|W| D); used only past
    # the SIZE CUTOFF.  Masks are neighbourhoods in the middle part.
    return [e for e in chunk if Xmask[e[0]] & Ymask[e[1]]]


class ExactTriangle:
    # Theorem 17 with D = 16 and g = 1: s = 4, pieces of C of 4 vertices, a
    # prime p in [sqrt(D)/2, sqrt(D)) = {2, 3}, middle parts C_k x Z_p of at
    # most 12 <= D vertices.  The theorem assumes D <= n and Theorem 19 solves
    # small instances by brute force; we run the reduction at every size.
    D = 16
    g = 1

    def __init__(self, oracle):
        self.oracle = oracle
        self.s = isqrt(self.D)
        self.primes = [p for p in range(2, self.s) if 2 * p >= self.s and all(p % d for d in range(2, p))]
        self.piece = -(-self.s // self.g)
        self.cache = {}

    def _cached(self, key, build):
        if key is None:
            return build()
        val = self.cache.get(key)
        if val is None:
            val = self.cache[key] = build()
        return val

    @staticmethod
    def _residue_bits(w, p):
        bits = []
        for row in w:
            r = [0] * p
            for c, x in enumerate(row):
                if x is not None:
                    r[x % p] |= 1 << c
            bits.append(r)
        return bits

    @staticmethod
    def _masks(w, p, c0, c1, shift, sign):
        out = []
        for row in w:
            m = 0
            for c in range(c0, c1):
                x = row[c]
                if x is not None:
                    m |= 1 << ((c - c0) * p + (sign * x + shift) % p)
            out.append(m)
        return out

    def zero_triangles(self, nA, nB, nC, ab, wAC, wBC, x_id=None, y_id=None):
        # ab: the A-B edges (a, b, w(a, b)) in row-major order; wAC / wBC: rows of
        # weights, None for a missing edge.  The paper gives a missing edge the
        # weight 3n^nu + 1, which lies in no zero triangle; dropping it instead
        # only removes false positives.  Returns every zero triangle (a, b, c).
        if not ab:
            return []
        # Hashing modulo a prime: F(p) + Z0 is the sum over the pairs of the
        # coefficient of x^(-w(a,b) mod p) in (PQ)[a, b] over Z[x]/(x^p - 1);
        # here each needed coefficient is read off residue bitsets directly.
        best = None
        for p in self.primes:
            Ab = self._cached(x_id and (x_id, "bits", p), lambda: self._residue_bits(wAC, p))
            Bb = self._cached(y_id and (y_id, "bits", p), lambda: self._residue_bits(wBC, p))
            cnt = 0
            for a, b, w in ab:
                ra, rb = Ab[a], Bb[b]
                for u in range(p):
                    cnt += _popcount(ra[u] & rb[(-w - u) % p])
            if best is None or cnt < best[0]:
                best = (cnt, p)
        p = best[1]
        # The instances: pieces C_k, W_rho cut into chunks of at most n^2/sqrt(D)
        # pairs, middle part C_k x Z_p with a ~ (c, sigma) iff
        # sigma = w(a,c) + rho and (c, sigma) ~ b iff sigma = -w(b,c) (mod p).
        pieces = [(c0, min(c0 + self.piece, nC)) for c0 in range(0, nC, self.piece)]
        n = max(nA, nB, nC)
        lim = max(1, n * n // self.s)
        W = [[] for _ in range(p)]
        for e in ab:
            W[e[2] % p].append(e)
        Xm = self._cached(x_id and (x_id, "masks", p), lambda: {
            (rho, k): self._masks(wAC, p, c0, c1, rho, 1)
            for rho in range(p) for k, (c0, c1) in enumerate(pieces)})
        Ym = self._cached(y_id and (y_id, "masks", p), lambda: [
            self._masks(wBC, p, c0, c1, 0, -1) for c0, c1 in pieces])
        instances = [(rho, k, Wr[i:i + lim])
                     for rho, Wr in enumerate(W) for i in range(0, len(Wr), lim)
                     for k in range(len(pieces))]
        # Non-adaptive: every instance exists before the first oracle call.
        answers = [self.oracle(Xm[rho, k], Ym[k], chunk,
                               x_id and (x_id, p, rho, k), y_id and (y_id, p, k))
                   for rho, k, chunk in instances]
        # Witnesses: scan the piece of every accepted pair.  Footnote 10's
        # all-edges version stops at a pair's first zero triangle; listing
        # needs every one, so each accepted piece is scanned to the end.
        found = []
        for (rho, k, _), acc in zip(instances, answers):
            c0, c1 = pieces[k]
            for a, b, w in acc:
                ra, rb = wAC[a], wBC[b]
                for c in range(c0, c1):
                    x, y = ra[c], rb[c]
                    if x is not None and y is not None and w + x + y == 0:
                        found.append((a, b, c))
        return found


def convolution_3sum(Xa, Ya, Za, et, x_id=None, y_id=None):
    # Convolution-3SUM -> Exact Triangle after [VW13] (reconstructed): all (i, j)
    # with Xa[i] + Ya[j] + Za[i + j] = 0, None marking a missing entry.  Write
    # i = i1 t + i2, j = j1 t + j2 and y = i2 + j2 in [0, 2t - 1).  Instance y
    # has parts A = {i1}, B = {j1}, C = {i2} with w(i1, i2) = Xa[i],
    # w(j1, i2) = Ya[j], w(i1, j1) = Za[i + j], so its zero triangles are exactly
    # the solutions with this y.  t = ceil(sqrt(len)): about 2 sqrt(n) instances
    # on sqrt(n) vertices per part, as in Theorem 21(a).
    ell = len(Xa)
    t = isqrt(ell - 1) + 1 if ell > 1 else 1
    q = -(-ell // t)
    wAC = et._cached(x_id and (x_id, "rows"), lambda: [
        [Xa[i1 * t + i2] if i1 * t + i2 < ell else None for i2 in range(t)]
        for i1 in range(q)])
    ab = [[] for _ in range(2 * t - 1)]
    for k, z in enumerate(Za):
        if z is None:
            continue
        y, s = k % t, k // t
        for y, s in ((y, s), (y + t, s - 1)):
            if y < 2 * t - 1 and 0 <= s < 2 * q - 1:
                for i1 in range(max(0, s - q + 1), min(q, s + 1)):
                    ab[y].append((i1, s - i1, z))
    pairs = []
    for y in range(2 * t - 1):
        if not ab[y]:
            continue
        ab[y].sort()
        wBC = et._cached(y_id and (y_id, y, "rows"), lambda: [
            [Ya[j1 * t + y - i2] if 0 <= y - i2 < t and j1 * t + y - i2 < ell else None
             for i2 in range(t)]
            for j1 in range(q)])
        for a, b, c in et.zero_triangles(q, q, t, ab[y], wAC, wBC,
                                          x_id, y_id and (y_id, y)):
            pairs.append((a * t + c, b * t + y - c))
    return pairs


def _primes_below(n):
    sieve = bytearray([1]) * max(n, 2)
    sieve[0] = sieve[1] = 0
    for i in range(2, isqrt(n - 1) + 1 if n > 1 else 0):
        if sieve[i]:
            sieve[i * i::i] = bytearray(len(sieve[i * i::i]))
    return [i for i in range(len(sieve)) if sieve[i]]


def three_sum_triples(S, et, tau=3):
    # 3SUM -> Convolution-3SUM.  The paper only cites [CH20] for this step; this
    # is our reconstruction in the style of [Pat10]: hash by the exactly linear
    # h(x) = x mod P into P >= n buckets, the prime P in [n, 2n) with the fewest
    # colliding pairs (chosen by counting, as [FKP24] chooses its modulus).
    # X_r[i] is the element of rank r in bucket i; a + b + c = 0 forces
    # X_r1[h(a)] + X_r2[h(b)] + Z_r3[h(a) + h(b)] = 0 with Z_r[k] = X_r[-k mod P],
    # one Convolution-3SUM instance per rank triple.  The three lists are one
    # set, so a solution can put its highest rank in Z and its lowest in Y: only
    # r3 >= r1 >= r2.  Elements of rank >= tau (overfull buckets, which [Pat10]
    # also leaves to brute force) are checked one by one in O(n) each.
    n = len(S)
    best = None
    for P in _primes_below(max(4, 2 * n)):
        if P < n:
            continue
        cnt = [0] * P
        for x in S:
            cnt[x % P] += 1
        coll = sum(c * (c - 1) for c in cnt)
        if best is None or coll < best[0]:
            best = (coll, P)
    P = best[1]
    buckets = [[] for _ in range(P)]
    for x in S:
        buckets[x % P].append(x)
    R = min(tau, max(len(b) for b in buckets))
    X = [[b[r] if r < len(b) else None for b in buckets] for r in range(R)]
    Z = [[Xr[-k % P] for k in range(2 * P - 1)] for Xr in X]
    for r3 in range(R):
        for r1 in range(r3 + 1):
            for r2 in range(r1 + 1):
                for i, j in convolution_3sum(X[r1], X[r2], Z[r3], et, ("X", r1), ("Y", r2)):
                    yield X[r1][i], X[r2][j], Z[r3][i + j]
    present = set(S)
    for b in buckets:
        for x in b[tau:]:
            for y in S:
                if -x - y in present:
                    yield x, y, -x - y


class Solution:
    def threeSum(self, nums: List[int]) -> List[List[int]]:
        # Listing: the reductions report every value triple (a, b, c) over the
        # distinct values with a + b + c = 0, and multiplicities are checked
        # after.  The output alone can have Theta(n^2) triplets (nums = -n..n),
        # so listing cannot be subquadratic in the worst case; the paper's bound
        # is for detection, and the witness scans here add time per solution.
        count = {}
        for x in nums:
            count[x] = count.get(x, 0) + 1
        S = sorted(count)
        if not S:
            return []
        oracle = Theorem5Oracle() if len(S) <= THEOREM5_MAX_DISTINCT else straightforward_oracle
        found, rejected = set(), set()
        for triple in three_sum_triples(S, ExactTriangle(oracle)):
            key = tuple(sorted(triple))
            if key in found or key in rejected:
                continue
            a, b, c = key
            if a == c:
                ok = count[a] >= 3
            elif a == b or b == c:
                ok = count[b] >= 2
            else:
                ok = True
            (found if ok else rejected).add(key)
        return [list(t) for t in sorted(found)]
