import Mathlib

/-!
# Verification for The Cost of Balance in Order-Sampled Model Comparisons

Checked with Lean 4.36.0 and Mathlib. The finite checks independently
enumerate displayed orders and balanced blocks and use exact integer
matrices. They use Lean's native evaluation for closed computations.
The analytical identities are ordinary kernel-checked proofs.
-/

namespace CostOfBalance

open scoped BigOperators

section VarianceIdentities

theorem weighted_second_moment {ι : Type*} [Fintype ι]
    (p x : ι → ℝ) (hp : ∑ i, p i = 1) (a : ℝ) :
    (∑ i, p i * (x i - a) ^ 2) =
      (∑ i, p i * x i ^ 2) - 2 * a * (∑ i, p i * x i) + a ^ 2 := by
  calc
    (∑ i, p i * (x i - a) ^ 2) =
        ∑ i, (p i * x i ^ 2 - (2 * a) * (p i * x i) + a ^ 2 * p i) := by
      apply Finset.sum_congr rfl
      intro i _
      ring
    _ = _ := by simp [Finset.sum_add_distrib, Finset.sum_sub_distrib,
      ← Finset.mul_sum, hp]

/-- A block's average squared error is its squared mean error plus
its within-block variance. This is the identity used in Theorem 2. -/
theorem block_variance_split {ι : Type*} [Fintype ι]
    (p x : ι → ℝ) (hp : ∑ i, p i = 1) (a : ℝ) :
    (∑ i, p i * (x i - a) ^ 2) =
      ((∑ i, p i * x i) - a) ^ 2 +
        (∑ i, p i * (x i - (∑ j, p j * x j)) ^ 2) := by
  rw [weighted_second_moment p x hp a,
    weighted_second_moment p x hp (∑ i, p i * x i)]
  ring

/-- Replacing a sample by its complement multiplies its squared
centered-mean error by the square of the ratio of their sizes. -/
theorem complement_scaling (k M b c : ℝ) (_hk : 0 < k) (hM : k < M)
    (h : k * b + (M - k) * c = 0) :
    c ^ 2 = (k / (M - k)) ^ 2 * b ^ 2 := by
  have hd : M - k ≠ 0 := ne_of_gt (sub_pos.mpr hM)
  have hc : c = -(k / (M - k)) * b := by
    have h' : c = -(k * b) / (M - k) := by
      apply (eq_div_iff hd).2
      linarith
    rw [h']
    ring
  rw [hc]
  ring

/-- The exact eight-call Williams-versus-precedence threshold. -/
theorem residual_threshold (eW eP N : ℝ) (hP : 0 < eP) (hN : 0 < N) :
    (2 / 5 : ℝ) * eW / (24 * N ^ 2) <
      (2 / 11 : ℝ) * eP / (24 * N ^ 2) ↔ eW / eP < 5 / 11 := by
  have hden : 0 < (24 : ℝ) * N ^ 2 := by positivity
  rw [div_lt_div_iff_of_pos_right hden, div_lt_iff₀ hP]
  constructor <;> intro h <;> linarith

/-- For integer dyad counts, their centered sum of squares is bounded
below by the floor/ceiling allocation. With the floor shift, h is the
remainder of the number of sampled orders modulo the number of dyads. -/
theorem dyad_count_bound {ι : Type*} [Fintype ι]
    (counts : ι → ℤ) (q : ℤ) (hcard : 0 < Fintype.card ι) :
    let m : ℝ := Fintype.card ι
    let z : ι → ℝ := fun i => (counts i - q : ℤ)
    let h := ∑ i, z i
    h * (m - h) ≤ m * ∑ i, (z i - h / m) ^ 2 := by
  dsimp only
  let m : ℝ := Fintype.card ι
  let z : ι → ℝ := fun i => (counts i - q : ℤ)
  let h := ∑ i, z i
  change h * (m - h) ≤ m * ∑ i, (z i - h / m) ^ 2
  have hm : 0 < m := by
    change (0 : ℝ) < (Fintype.card ι : ℝ)
    exact_mod_cast hcard
  have hs : h ≤ ∑ i, (z i) ^ 2 := by
    apply Finset.sum_le_sum
    intro i _
    have hz : counts i - q ≤ 0 ∨ 1 ≤ counts i - q := by omega
    rcases hz with hz | hz
    · have hr : z i ≤ 0 := by
        change ((counts i - q : ℤ) : ℝ) ≤ 0
        exact_mod_cast hz
      exact hr.trans (sq_nonneg _)
    · have hr : 1 ≤ z i := by
        change 1 ≤ ((counts i - q : ℤ) : ℝ)
        exact_mod_cast hz
      nlinarith [mul_nonneg (sub_nonneg.mpr hr) (show 0 ≤ z i by linarith)]
  have hv : (∑ i, (z i - h / m) ^ 2) =
      (∑ i, (z i) ^ 2) - h ^ 2 / m := by
    calc
      (∑ i, (z i - h / m) ^ 2) =
          ∑ i, ((z i) ^ 2 - (2 * (h / m)) * z i + (h / m) ^ 2) := by
        apply Finset.sum_congr rfl
        intro i _
        ring
      _ = (∑ i, (z i) ^ 2) - (2 * (h / m)) * h + m * (h / m) ^ 2 := by
        simp [Finset.sum_add_distrib, Finset.sum_sub_distrib,
          ← Finset.mul_sum, nsmul_eq_mul, h, m]
      _ = _ := by
        field_simp [ne_of_gt hm]
        ring
  calc
    h * (m - h) = m * h - h ^ 2 := by ring
    _ ≤ m * (∑ i, (z i) ^ 2) - h ^ 2 :=
      sub_le_sub_right (mul_le_mul_of_nonneg_left hs hm.le) _
    _ = _ := by
      rw [hv]
      field_simp [ne_of_gt hm]

/-- The two swap-witness risks have the position lower bound as their
minimum envelope, with equality only at the stated swap target. -/
theorem swap_witness_envelope (n t : ℝ) (hn : 4 ≤ n) :
    let s := n * (n - 3) / 2
    let a := (n - 1) * (n - 2) / 2
    let R := 1 / n + (n - 1) / n * max (t / s) ((1 - t) / a)
    (n - 2) / (n ^ 2 - 3 * n + 1) ≤ R ∧
      (R = (n - 2) / (n ^ 2 - 3 * n + 1) ↔ t = s / (s + a)) := by
  dsimp only
  let s := n * (n - 3) / 2
  let a := (n - 1) * (n - 2) / 2
  let M := max (t / s) ((1 - t) / a)
  change (n - 2) / (n ^ 2 - 3 * n + 1) ≤ 1 / n + (n - 1) / n * M ∧
    (1 / n + (n - 1) / n * M = (n - 2) / (n ^ 2 - 3 * n + 1) ↔
      t = s / (s + a))
  have hn0 : 0 < n := by linarith
  have hs : 0 < s := by
    dsimp [s]
    exact div_pos (mul_pos hn0 (by linarith)) (by norm_num)
  have ha : 0 < a := by
    dsimp [a]
    exact div_pos (mul_pos (by linarith) (by linarith)) (by norm_num)
  have hsa : 0 < s + a := add_pos hs ha
  have hd : s + a = n ^ 2 - 3 * n + 1 := by dsimp [s, a]; ring
  have hfactor : 0 < (n - 1) / n := div_pos (by linarith) hn0
  have h1 : t ≤ M * s := (div_le_iff₀ hs).mp (le_max_left _ _)
  have h2 : 1 - t ≤ M * a := (div_le_iff₀ ha).mp (le_max_right _ _)
  have hm : 1 / (s + a) ≤ M := (div_le_iff₀ hsa).mpr (by nlinarith [h1, h2])
  have hformula : (n - 2) / (n ^ 2 - 3 * n + 1) =
      1 / n + (n - 1) / n * (1 / (s + a)) := by
    rw [← hd]
    field_simp [ne_of_gt hn0, ne_of_gt hsa]
    nlinarith [hd]
  rw [hformula]
  refine ⟨add_le_add (le_refl _) (mul_le_mul_of_nonneg_left hm hfactor.le), ?_⟩
  constructor
  · intro h
    have he : M = 1 / (s + a) := by
      exact (mul_left_cancel₀ (ne_of_gt hfactor)) (add_left_cancel h)
    have ht₁ : t * (s + a) ≤ s := by
      have := (mul_le_mul_of_nonneg_right h1 hsa.le)
      rw [he] at this
      field_simp [ne_of_gt hsa] at this
      nlinarith
    have ht₂ : (1 - t) * (s + a) ≤ a := by
      have := (mul_le_mul_of_nonneg_right h2 hsa.le)
      rw [he] at this
      field_simp [ne_of_gt hsa] at this
      nlinarith
    apply (eq_div_iff (ne_of_gt hsa)).2
    nlinarith
  · intro ht
    have hleft : t / s = 1 / (s + a) := by
      rw [ht]
      field_simp [ne_of_gt hs, ne_of_gt hsa]
    have hright : (1 - t) / a = 1 / (s + a) := by
      rw [ht]
      field_simp [ne_of_gt ha, ne_of_gt hsa]
      ring
    simp [M, hleft, hright]

end VarianceIdentities

namespace FourOptions

abbrev Block := UInt32
abbrev Matrix := Array (Array Int)

def labels : List Nat := List.range 4
def rowIndices : List Nat := List.range 24

def orders : Array (Array Nat) := Id.run do
  let mut rows := #[]
  for a in labels do
    for b in labels do
      for c in labels do
        for d in labels do
          if a != b && a != c && a != d && b != c && b != d && c != d then
            rows := rows.push #[a, b, c, d]
  return rows

def bit (i : Nat) : Block := (1 : UInt32) <<< UInt32.ofNat i
def includes (b : Block) (i : Nat) : Bool := b &&& bit i != 0

def block (indices : List Nat) : Block :=
  indices.foldl (fun b i => b ||| bit i) 0

def indices (b : Block) : List Nat := rowIndices.filter (includes b)

def counts6 : Array Nat :=
  (List.range 64).toArray.map fun m =>
    ((List.range 6).filter fun i => includes (UInt32.ofNat m) i).length

def size (b : Block) : Nat :=
  counts6[(b &&& 63).toNat]! + counts6[((b >>> 6) &&& 63).toNat]! +
    counts6[((b >>> 12) &&& 63).toNat]! + counts6[((b >>> 18) &&& 63).toNat]!

def slot (row : Array Nat) (label : Nat) : Nat :=
  (labels.find? fun i => row[i]! == label).getD 4

def feature (p : Array Nat → Bool) : Block :=
  block (rowIndices.filter fun i => p orders[i]!)

def positionFeatures : List Block :=
  labels.flatMap fun s => labels.map fun a => feature (fun row => row[s]! == a)
def precedenceFeatures : List Block :=
  labels.flatMap fun a => (labels.filter (a < ·)).map fun b =>
    feature (fun row => slot row a < slot row b)
def adjacencyFeatures : List Block :=
  labels.flatMap fun a => (labels.filter (a != ·)).map fun b =>
    feature (fun row => slot row a + 1 == slot row b)

def parts (k a : Nat) : List Block :=
  ((List.range 64).map UInt32.ofNat).filterMap fun b =>
    if size b == k / 4 then some (b <<< UInt32.ofNat (6 * a)) else none

structure Catalogue where
  position : Array Block
  precedence : Array Block
  adjacency : Array Block
  deriving Inhabited

def catalogue (k : Nat) : Catalogue := Id.run do
  let mut position := #[]
  let mut precedence := #[]
  let mut adjacency := #[]
  let pos := positionFeatures
  let pre := precedenceFeatures
  let adj := adjacencyFeatures
  for a in parts k 0 do
    for b in parts k 1 do
      for c in parts k 2 do
        for d in parts k 3 do
          let mask := a ||| b ||| c ||| d
          if pos.all (fun f => size (mask &&& f) == k / 4) then
            position := position.push mask
            if pre.all (fun f => size (mask &&& f) == k / 2) then
              precedence := precedence.push mask
              if adj.all (fun f => size (mask &&& f) == k / 4) then
                adjacency := adjacency.push mask
  return ⟨position, precedence, adjacency⟩

def rowIndex (row : Array Nat) : Nat :=
  (rowIndices.find? fun i => orders[i]! == row).getD 24

def transform (b : Block) (relabel perm : Array Nat) : Block :=
  block ((indices b).map fun i =>
    let row := orders[i]!
    rowIndex (perm.map fun s => relabel[row[s]!]!))

def orbit (b : Block) (kind : Nat) : List Block :=
  let permutations := if kind == 0 then orders.toList else [orders[0]!, orders[23]!]
  (orders.toList.flatMap fun relabel =>
    permutations.map (transform b relabel)).eraseDups

def atoms (entries : List (List Nat × Nat)) : List (Block × Nat) :=
  entries.map fun (rows, weight) => (block rows, weight)

def templates (k kind : Nat) : List (Block × Nat) := atoms (match k, kind with
  | 4, 0 => [([0,7,17,22], 1)]
  | 4, 1 => [([0,7,17,22], 1), ([0,10,13,23], 1)]
  | 4, 2 => [([0,10,13,23], 1)]
  | 8, 0 => [([0,1,8,9,16,17,18,19], 6), ([0,3,7,11,12,16,20,23], 1)]
  | 8, 1 => [([0,1,6,10,13,17,22,23], 2), ([0,2,7,10,13,17,21,22], 2),
      ([0,3,7,11,12,16,20,23], 1), ([0,3,7,11,14,17,18,22], 6)]
  | 8, 2 => [([0,1,8,10,13,17,19,23], 4), ([0,2,7,10,13,16,21,23], 1)]
  | 12, 0 => [([0,1,2,6,7,10,15,16,17,19,22,23], 5),
      ([0,1,2,6,9,10,13,16,17,18,21,22], 4),
      ([0,1,2,6,9,10,13,16,17,19,20,23], 26)]
  | 12, 1 => [([0,1,2,7,8,10,13,16,17,19,21,23], 15),
      ([0,1,2,7,8,11,13,16,17,19,20,23], 30),
      ([0,1,3,6,7,11,14,16,17,18,22,23], 21),
      ([0,1,3,6,10,11,13,14,17,18,22,23], 12),
      ([0,2,4,7,8,11,13,15,17,19,20,22], 30),
      ([0,3,4,7,8,11,12,15,16,19,20,23], 2)]
  | 12, 2 => [([0,1,2,7,8,10,13,16,17,19,21,23], 3),
      ([0,1,3,8,10,11,13,14,17,18,19,23], 2),
      ([0,3,4,7,8,11,12,15,16,19,20,23], 1)]
  | _, _ => [])

def law (k kind : Nat) : Nat × List (Block × Nat) :=
  let orbits := (templates k kind).map fun (b, w) => (orbit b kind, w)
  let multiple := orbits.foldl (fun n (bs, _) => Nat.lcm n bs.length) 1
  let totalWeight := (orbits.map Prod.snd).sum
  let weightedBlocks := orbits.flatMap fun (bs, w) =>
    bs.map fun b => (b, w * (multiple / bs.length))
  (totalWeight * multiple, weightedBlocks)

def matrix (f : Nat → Nat → Int) : Matrix :=
  rowIndices.toArray.map fun i => rowIndices.toArray.map (f i)

def entry (a : Matrix) (i j : Nat) : Int := (a[i]!)[j]!

def multiply (a b : Matrix) : Matrix :=
  matrix fun i j => (rowIndices.map fun t => entry a i t * entry b t j).sum

def matrixEvery (f : Nat → Nat → Bool) : Bool :=
  rowIndices.all fun i => rowIndices.all (f i)

def covariance (k n : Nat) (weights : List (Block × Nat)) : Matrix :=
  matrix fun i j =>
    let count := (weights.map fun (b, w) => if includes b i && includes b j then w else 0).sum
    576 * (count : Int) - (k : Int) ^ 2 * (n : Int)

def validCovariance (k n : Nat) (a : Matrix) (weights : List (Block × Nat))
    (eligible : Array Block) : Bool :=
  n > 0 && (weights.map Prod.snd).sum == n &&
    weights.all (fun (b, w) => w > 0 && size b == k && eligible.contains b) &&
    rowIndices.all (fun i =>
      let marginal := (weights.map fun (b, w) => if includes b i then w else 0).sum
      24 * marginal == k * n && a[i]!.toList.sum == 0) &&
    (k : Int) * (rowIndices.map fun i => entry a i i).sum ==
      (24 - (k : Int)) * (24 * (k : Int) ^ 2 * (n : Int))

def tightFrame (k : Nat) (a : Matrix) (den p q : Int)
    (eligible : Array Block) : Bool :=
  let squared := multiply a a
  matrixEvery (fun i j => q * entry squared i j == p * den * entry a i j) &&
    eligible.all fun b =>
      let rows := indices b
      rowIndices.all fun i =>
        24 * q * (rows.map fun j => entry a i j).sum ==
        p * den * ((if includes b i then 24 else 0) - (k : Int))

def inverseIndex (i : Nat) : Nat :=
  rowIndex (labels.toArray.map (slot orders[i]!))

def displayMask (rows : List Nat) : Block := block (rows.map inverseIndex)

def witnessLower (k kind : Nat) (eligible : Array Block) (p q : Int)
    (positive negative : Block) : Bool :=
  let norm : Int := size positive + size negative
  norm > 0 && (positive &&& negative) == 0 && size positive == size negative &&
    eligible.all fun b =>
      let bs := orbit b kind
      let riskNumerator := (bs.map fun c =>
        ((size (c &&& positive) : Int) - (size (c &&& negative) : Int)) ^ 2).sum
      q * 24 * riskNumerator ≥ p * (k : Int) ^ 2 * (bs.length : Int) * norm

def spectralBound (a : Matrix) (den : Int) (roots : List (Int × Int)) : Bool :=
  let identity := matrix fun i j => if i == j then 1 else 0
  let product := roots.foldl (fun acc (p, q) =>
    multiply acc (matrix fun i j => q * entry a i j - if i == j then p * den else 0)) identity
  matrixEvery fun i j => entry product i j == 0

def fourCallChecks (data : Catalogue) : Bool :=
  let (n₁, w₁) := law 4 0
  let (n₂, w₂) := law 4 1
  let (n₃, w₃) := law 4 2
  let a₁ := covariance 4 n₁ w₁
  let a₂ := covariance 4 n₂ w₂
  let a₃ := covariance 4 n₃ w₃
  validCovariance 4 n₁ a₁ w₁ data.position &&
    validCovariance 4 n₂ a₂ w₂ data.precedence &&
    validCovariance 4 n₃ a₃ w₃ data.adjacency &&
    spectralBound a₁ (384 * (n₁ : Int)) [(0,1), (1,3), (1,2)] &&
    spectralBound a₂ (384 * (n₂ : Int)) [(0,1), (1,4), (1,2), (3,4)] &&
    tightFrame 4 a₃ (384 * (n₃ : Int)) 1 1 data.adjacency &&
    witnessLower 4 0 data.position 1 2 (displayMask [0,7,16,23]) (displayMask [3,11,12,20]) &&
    witnessLower 4 1 data.precedence 3 4 (displayMask [0,2,7,10,13,16,21,23])
      (displayMask [3,5,9,11,12,14,18,20])

def frameCheck (k kind : Nat) (eligible : Array Block) (p q : Int) : Bool :=
  let (n, weights) := law k kind
  let a := covariance k n weights
  validCovariance k n a weights eligible &&
    tightFrame k a (24 * (k : Int) ^ 2 * (n : Int)) p q eligible

def eightCallChecks (data : Catalogue) : Bool :=
  frameCheck 8 0 data.position 1 7 &&
    frameCheck 8 1 data.precedence 2 11 &&
    frameCheck 8 2 data.adjacency 2 5

def twelveCallChecks (data : Catalogue) : Bool :=
  frameCheck 12 0 data.position 1 14 &&
    frameCheck 12 1 data.precedence 1 11 &&
    frameCheck 12 2 data.adjacency 1 6

def checkCount (data : Catalogue) (p a f : Nat) : Bool :=
  data.position.size == p && data.precedence.size == a && data.adjacency.size == f

def williamsPairsCheck : Bool :=
  let williams := (catalogue 4).adjacency.toList
  let pairs := williams.flatMap fun a =>
    (williams.filter (a < ·)).map fun b => a ||| b
  let blocks := (catalogue 8).adjacency
  let (n₄, w₄) := law 4 2
  let (n₈, w₈) := law 8 2
  let a₄ := covariance 4 n₄ w₄
  let a₈ := covariance 8 n₈ w₈
  let d₄ : Int := 384 * n₄
  let d₈ : Int := 1536 * n₈
  williams.length == 6 && (williams.map size).sum == 24 &&
    size (williams.foldl (· ||| ·) 0) == 24 &&
    pairs.eraseDups.length == 15 && blocks.size == 15 &&
    pairs.all (fun b => blocks.contains b &&
      15 * (w₈.map (fun (c, w) => if c == b then w else 0)).sum == n₈) &&
    matrixEvery (fun i j => 5 * d₄ * entry a₈ i j == 2 * d₈ * entry a₄ i j)

def residualDirections : List (Block × Block) := [
  (block [0,7,16,23], block [2,10,13,21]),
  (block [0,7,17,22], block [1,6,16,23]),
  (block [1,6,17,22], block [4,8,15,19]),
  (block [2,11,13,20], block [3,10,12,21]),
  (block [3,11,12,20], block [5,9,14,18]),
  (block [4,9,14,19], block [5,8,15,18])]

def signedCount (p n f : Block) : Int := (size (p &&& f) : Int) - size (n &&& f)

def sparsePilotCheck : Bool :=
  let williams := (catalogue 4).adjacency.toList
  let pairs := williams.flatMap fun a =>
    (williams.filter (a < ·)).map fun b => a ||| b
  let residual := residualDirections.map fun (p, n) => p ||| n
  let features := positionFeatures ++ precedenceFeatures ++ williams
  residualDirections.length == 6 &&
    residualDirections.all (fun (p, n) =>
      size p == 4 && size n == 4 && (p &&& n) == 0 &&
      features.all (fun f => signedCount p n f == 0) &&
      residualDirections.all fun (q, m) =>
        signedCount p n q - signedCount p n m ==
          (if (p, n) == (q, m) then 8 else 0)) &&
    (pairs ++ residual).eraseDups.length == 18 &&
    rowIndices.all fun i =>
      2 * (pairs.filter (fun b => includes b i)).length +
        5 * (residual.filter (fun b => includes b i)).length == 20

theorem four_call_certificate :
    (let data := catalogue 4
     checkCount data 24 18 6 && fourCallChecks data) = true := by native_decide

theorem eight_call_certificate :
    (let data := catalogue 8
     checkCount data 255 99 15 && eightCallChecks data) = true := by native_decide

theorem twelve_call_certificate :
    (let data := catalogue 12
     checkCount data 640 180 22 && twelveCallChecks data) = true := by native_decide

theorem williams_pairs_certificate : williamsPairsCheck = true := by native_decide

theorem sparse_pilot_geometry_certificate : sparsePilotCheck = true := by native_decide

theorem complementary_risk_table :
    (8 / 16 : ℚ) ^ 2 * (1 / 7) = 1 / 28 ∧
    (8 / 16 : ℚ) ^ 2 * (2 / 11) = 1 / 22 ∧
    (8 / 16 : ℚ) ^ 2 * (2 / 5) = 1 / 10 ∧
    (4 / 20 : ℚ) ^ 2 * (1 / 2) = 1 / 50 ∧
    (4 / 20 : ℚ) ^ 2 * (3 / 4) = 3 / 100 ∧
    (4 / 20 : ℚ) ^ 2 * 1 = 1 / 25 := by norm_num

end FourOptions
end CostOfBalance

#print axioms CostOfBalance.block_variance_split
#print axioms CostOfBalance.complement_scaling
#print axioms CostOfBalance.residual_threshold
#print axioms CostOfBalance.dyad_count_bound
#print axioms CostOfBalance.swap_witness_envelope
#print axioms CostOfBalance.FourOptions.four_call_certificate
#print axioms CostOfBalance.FourOptions.eight_call_certificate
#print axioms CostOfBalance.FourOptions.twelve_call_certificate
#print axioms CostOfBalance.FourOptions.williams_pairs_certificate
#print axioms CostOfBalance.FourOptions.sparse_pilot_geometry_certificate
#print axioms CostOfBalance.FourOptions.complementary_risk_table
