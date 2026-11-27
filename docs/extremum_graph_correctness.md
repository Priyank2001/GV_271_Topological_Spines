# Correctness of the saddle computation

This note proves what steps 1–4 of `pipeline/extremum_graph.py` compute:
the maxima, their regions, the saddles, and the peak persistence. It states
each claim as a theorem about a finite graph, so it holds for every
`NeighborGraph` (kNN, grids, Freudenthal grids, in any dimension). Each
theorem is checked against a slow, independent computation in
`tests/test_extremum_graph_proof.py`.

What the proofs do **not** cover is listed at the end (loop filter,
cancellation order, saturation, and closeness to the continuous field).

## Setting and notation

* $G = (V, E)$ is a finite, simple, undirected graph with $N = |V|$
  vertices (`NeighborGraph`: `edges` holds each edge once, `indptr`/`indices`
  is the symmetric adjacency). $G$ need not be connected.
* $f : V \to \mathbb{R}$ is the sampled field (`values`).
* **Total order.** `total_order` gives each vertex a rank
  $r(v) \in \{0, \dots, N-1\}$, sorting by $(f(v), v)$. $r$ is a bijection,
  and $r(u) > r(v) \Rightarrow f(u) \ge f(v)$. We write $u \succ v$ for
  $r(u) > r(v)$ and say $u$ is *higher*. All proofs use $\succ$ only, so ties
  in $f$ need no special care.
* $N(v)$ is the set of neighbours of $v$, and $\mathrm{Lk}^+(v) = \{u \in N(v) : u \succ v\}$
  its *upper link*.
* For a threshold $t$, the **superlevel graph** is
  $G_t = G[\{v : r(v) \ge t\}]$, the subgraph induced on the vertices of rank at least $t$.
  $G_{>t}$ is defined the same way with $>$.
* A *path* is a sequence of vertices with consecutive ones adjacent. Its
  *bottleneck* is its lowest vertex (w.r.t. $\succ$).

## Step 2: maxima and regions

Define $\mathrm{up}(v)$ as the highest vertex in $N(v)$ if that vertex is higher
than $v$, and $v$ itself otherwise (`steepest_ascent`; a vertex of degree 0
points to itself). A **maximum** is a vertex with $\mathrm{up}(v) = v$, i.e. a vertex
with $\mathrm{Lk}^+(v) = \emptyset$.

**Lemma A (ascent ends at a maximum).** For every $v$, the sequence
$v, \mathrm{up}(v), \mathrm{up}^2(v), \dots$ reaches a maximum $M(v)$ after
fewer than $N$ steps and stays there. `_pointer_jump` returns $M$.

*Proof.* While $\mathrm{up}(x) \ne x$ the rank strictly increases, so no vertex
repeats and the sequence becomes stationary within $N - 1$ steps, at a
vertex with $\mathrm{up}(x) = x$, i.e. a maximum. `_pointer_jump` replaces
$\mathrm{up}$ by $\mathrm{up} \circ \mathrm{up}$ until nothing changes. After $j$ rounds every pointer
equals $\mathrm{up}^{2^j}$, which is $M$ once $2^j \ge N - 1$. Before that, the
array can stop changing only if every pointer is already a fixed point, which
is again $M$. $\square$

The **region** of a maximum $m$ is $R(m) = \{v : M(v) = m\}$ (`region`). The
regions partition $V$ and $m \in R(m)$.

**Lemma B (ascent paths stay in the region and go up).** If $v \in R(m)$,
every vertex $x$ on the ascent path $v, \mathrm{up}(v), \dots, m$ lies in $R(m)$ and
satisfies $x \succeq v$.

*Proof.* The ascent path from $x$ is a suffix of the one from $v$, so
$M(x) = m$. Ranks increase along the path. $\square$

**Corollary C.** For every $t$, $R(m) \cap \{r \ge t\}$ is empty or connected in $G[R(m)]$
and contains $m$. In particular every region is connected, and every
$v$ is joined to $M(v)$ by a path in $G_{r(v)}$.

*Proof.* If $v \in R(m)$ has $r(v) \ge t$, its ascent path joins it to $m$ inside
$R(m) \cap \{r \ge t\}$ by Lemma B. $\square$

## Step 3: saddles

Call an edge $\{u, w\}$ a **cross edge** between regions $A \ne B$ if one end
is in $A$ and the other in $B$. Its *level* is $\min(r(u), r(w))$, and its
*lower endpoint* is the end with that rank. For every pair of regions joined by
at least one cross edge, `passes` picks a cross edge of highest level and
reports its lower endpoint as the **saddle** $\sigma(A, B)$.

*Well defined.* The maximum level over a finite set of edges is attained.
Because $r$ is a bijection, all cross edges at that level share the same lower
endpoint, so $\sigma(A, B)$ does not depend on which of them is picked.
(The code sorts by (pair, level) and takes the last edge of each pair group.)

**Theorem 1 (each saddle has degree 2).** Each saddle record joins exactly
two distinct maxima, and no pair of maxima has more than one record. Hence, in the
extremum graph (maxima and saddles as nodes, `edges` as links), every saddle
node has degree exactly 2. This stays true after the loop filter (step 5) and
after `simplify`.

*Proof.* `passes` keeps only edges with `region[u] != region[w]`, and it
groups them by the unordered pair $(\min, \max)$ of region ids, emitting one
record per group. So the two ids differ and no pair repeats. Step 5 only
deletes records. `simplify` maps both ends to surviving peaks, deletes a
record whose ends coincide (`sp[:, 0] != sp[:, 1]`), and then keeps only the
first (highest) record of each remaining pair. `edges` emits one link per
end. $\square$

Maxima, on the other hand, have unbounded degree: one link per adjacent region.
A *vertex* can still carry several saddle records (one per pair it is the
pass of), e.g. where three basins meet. See Corollary 5.

**Theorem 2 (the saddle is the best pass inside $A \cup B$).** Let $a, b$ be the
maxima of adjacent regions $A, B$ and $s = \sigma(A, B)$. Among all paths from
$a$ to $b$ in $G[A \cup B]$, the highest bottleneck is $s$:
$$
r(s) = \max_{P : a \leadsto b \text{ in } G[A \cup B]} \; \min_{x \in P} r(x).
$$

*Proof.* ($\le$) Let $\{s, u\}$ be the chosen cross edge, so $u \succ s$, and
say $s \in A$, $u \in B$ (the other case is symmetric). By Lemma B, the ascent path from $s$ to
$a$ stays in $A$ with ranks $\ge r(s)$, and the ascent path from $u$ to $b$
stays in $B$ with ranks $\ge r(u) > r(s)$. Joining them through the edge
$\{s, u\}$ gives a path in $G[A \cup B]$ with bottleneck exactly $s$.

($\ge$) Any path from $a \in A$ to $b \in B$ inside $A \cup B$ has a step that
leaves $A$ for $B$, that is, a cross edge between $A$ and $B$. Its bottleneck is at most
that edge's level, which is at most $r(s)$ by the choice of $s$. $\square$

So $s$ is the vertex at which $a$ and $b$ first become connected when the
vertices of $A \cup B$ are added from the top down. Note the restriction
to $A \cup B$: a path through a third region may have a higher
bottleneck. That case is a *loop-closing* saddle in step 4.

## Step 4: persistence

Let $H$ be the **region graph**: maxima as nodes, one arc $\{a, b\}$ per saddle
record, weighted by $r(\sigma)$. Let $H_t$ be its subgraph on the maxima with
$r \ge t$ and the arcs with $r(\sigma) \ge t$. (An arc's two maxima are higher
than its saddle, so $H_t$ is a graph.)

**Theorem 3 (superlevel components = region-graph components).** For every
$t$:

1. every connected component of $G_t$ contains a maximum, and
2. two maxima $a, b$ with $r \ge t$ lie in the same component of $G_t$ if and
   only if they lie in the same component of $H_t$.

So $C \mapsto \{\text{maxima in } C\}$ is a bijection from the components of $G_t$ to
those of $H_t$.

*Proof.* (1) By Corollary C, any $v \in G_t$ is joined to $M(v)$ inside
$G_{r(v)} \subseteq G_t$.

(2, $\Leftarrow$) It suffices to treat one arc $\{a, b\}$ of $H_t$. The path
built in Theorem 2 has bottleneck $\sigma(a, b)$, so all its vertices have rank
$\ge r(\sigma) \ge t$ and it lies in $G_t$.

(2, $\Rightarrow$) Let $a = p_0, p_1, \dots, p_k = b$ be a path in $G_t$. Then
$M(p_i) \succeq p_i$, so every $M(p_i)$ is a node of $H_t$. When
$M(p_i) \ne M(p_{i+1})$, the edge $\{p_i, p_{i+1}\}$ is a cross edge between
their regions at level $\ge t$. The pair's saddle is at least as high, so
$\{M(p_i), M(p_{i+1})\}$ is an arc of $H_t$. Dropping repeats, $M(p_0), \dots, M(p_k)$
is a walk from $a$ to $b$ in $H_t$. $\square$

**Persistence of a superlevel filtration** (the reference we compare with).
Add the vertices of $G$ from highest to lowest. A vertex with no added
neighbour is a maximum and starts a new component. When vertex $v$ joins
components $C_1, \dots, C_j$ ($j \ge 2$), the component with the highest
maximum survives (*elder rule*). Each of the other $j - 1$ components dies, and
its maximum $m$ gets persistence $f(m) - f(v)$. Maxima whose component never
dies (one per connected component of $G$) get $\infty$.

**Theorem 4 (`_sweep` computes this persistence).** `_sweep` gives every maximum
the persistence above. A saddle record is marked `merge` exactly when it
reduces the number of union-find groups, and for every vertex $v$
$$
\#\{\text{merge records at } v\} = (\text{number of components of } G_{>r(v)} \text{ that } v \text{ touches}) - 1 \quad (\text{or } 0 \text{ if } v \text{ touches none}).
$$
So the merge saddles, counted with multiplicity, are exactly the merge events of
the superlevel filtration.

*Proof.* The sweep visits records by decreasing $r(\sigma)$ (`passes` returns
them sorted that way), unioning the two maxima of each record. `top[root]` is
the highest maximum of each group: initially true, and preserved because a union keeps the
group whose top has the smaller id, i.e. the higher peak.

*Invariant.* After all records with $r(\sigma) \ge t$ are processed, the groups
of the maxima with $r \ge t$ are the components of $H_t$. The maxima below $t$ are
untouched singletons, because no processed arc has an end below $t$. By
Theorem 3 these groups are the components of $G_t$.

Now fix a vertex $v$ and set $t = r(v)$. The partition into components changes between
$G_{>t}$ and $G_t$ only by $v$ joining the components it touches. A
maximum's rank is never a saddle's rank (a saddle has a higher neighbour), so
the records at level $t$ are exactly those with $\sigma = v$. By the
invariant, processing them turns the groups of $G_{>t}$ into the groups
of $G_t$. If $v$ touches $j$ components, they become one group, so exactly
$j - 1$ of these records reduce the group count. Those are the `merge`
records, which proves the count. The others join maxima already in one group and are
loop-closing.

*Which maxima die.* Within one merge event, every union kills the group
whose top is lower. A top dies at most once, the highest top among the $j$
groups never dies, and there are $j - 1$ deaths. So the tops that die are
exactly the $j - 1$ non-eldest ones, *whatever order* the tied records are
processed in. Each gets `peak_vals[top] - sad_vals[s]` $= f(m) - f(v)$, the
elder-rule value. A top that never dies keeps $\infty$, and by Theorem 3 at the lowest
level those are one per component of $G$. $\square$

The loop filter (step 5) runs after `_sweep` and keeps every merge record
(`keep = is_merge | ...`), so Theorem 4 holds for the returned graph whatever
`min_loop_relief` is.

**Corollary 5 (merge saddles are true graph saddles).** If $v$ carries a
merge record, then its upper link $G[\mathrm{Lk}^+(v)]$ is disconnected. If $v$ carries
$\mu$ merge records, its higher neighbours lie in $\mu + 1$ distinct components of
$G_{>r(v)}$.

*Proof.* By Theorem 4, $v$ touches $\mu + 1 \ge 2$ components of $G_{>r(v)}$
through neighbours in $\mathrm{Lk}^+(v)$. A path inside $G[\mathrm{Lk}^+(v)]$ lies in
$G_{>r(v)}$, so neighbours in different components of $G_{>r(v)}$ cannot be
connected in the upper link. $\square$

The converse direction, "every vertex where superlevel components merge
carries a merge record", is the count formula in Theorem 4 with $j \ge 2$.
So the merge saddles are complete as well as sound. A loop-closing
saddle, by contrast, is only guaranteed the weaker property of Theorem 2 (a
disconnected upper link *within* $A \cup B$). This is why step 5 needs a
heuristic to decide which of them to keep.

## Summary

| Claim | Statement | Where |
|---|---|---|
| Lemma A | ascent terminates at a maximum; pointer jumping computes it | `steepest_ascent`, `_pointer_jump` |
| Cor. C | regions are connected, and so is every superlevel slice of a region | |
| Thm 1 | every saddle joins exactly 2 distinct maxima, at most one per pair, also after filtering and `simplify` | `passes`, `simplify`, `edges` |
| Thm 2 | $\sigma(A, B)$ is the bottleneck of the best $a \leadsto b$ path inside $A \cup B$ | `passes` |
| Thm 3 | components of $G_t$ ↔ components of $H_t$ | |
| Thm 4 | `peak_persistence` = 0-dim superlevel persistence (elder rule); merge records = merge events, with multiplicity | `_sweep` |
| Cor. 5 | a merge saddle has a disconnected upper link | |

## Not covered

* **Loop filter (step 5).** Which loop-closing saddles are real is a
  question about the underlying continuous field, not about $G$, and the
  relief test is a heuristic. The proofs only use the fact that it never drops a merge
  record.
* **`_cancellations` and `_saturation` (step 6).** These replay Correa et al.
  (2011), Algorithm 1. `simplify` reads persistence from `_sweep` and parents from
  `_cancellations`. That the two agree on which peak dies at each merge
  saddle is checked by tests (`test_a_cancelled_peak_joins_the_peak_across_its_saddle`,
  the saturation tests), not proved here.
* **One saddle per pair.** If two maxima are separated by two distinct
  passes, only the higher is reported (Theorem 2 picks the best path). This is a
  definition, not an error, but it means the extremum graph can have fewer
  loops than the continuous field.
* **Sampling.** All statements are about $f$ on $G$. Whether the graph's maxima
  and saddles match those of the continuous field depends on the
  neighbourhood graph (choice of $k$, Freudenthal vs. axis grid) and the
  sampling density.
* **Backends.** The proofs are about the NumPy backend. The PyTorch backend is
  checked to give identical output in `tests/test_backends.py`.
