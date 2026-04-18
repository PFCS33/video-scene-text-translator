# [Me]:

I integrated the trained refinement model into the pipeline. It does not perform very well, although the visualizations of the network alone look great. Sometimes it even adds jitter/drift to an already well-aligned ROI. I have another idea: how about we "cross-reference" the predicted delta_H? For each source-target/reference ROI pair, we randomly pick some frames i on the same track, predict source-to-i and i-to-reference delta_H, multiply them to get more source-to-target delta_Hs, and calculate a weighted average between those new delta_Hs and the original delta_H. Does this idea make sense?


# [ChatGPT]:

Yes, the idea makes sense in spirit, but **not as a plain weighted average of homography matrices**.

The good part of your idea is this:

* you suspect the per-pair prediction is noisy
* you want to use **other frames on the same track as auxiliary evidence**
* you want to exploit the fact that alignment along a track should be **mutually consistent**

That is exactly the right instinct. In multi-frame alignment, composing inter-frame homographies to obtain additional estimates is standard, and using all available inter-frame relations is usually better than trusting one noisy estimate alone. ([ResearchGate][1])

But the risky part is this:

* homographies live in a **projective transformation space**
* naïvely averaging their matrix entries is generally not geometrically meaningful

Homographies are only defined up to scale, and the space is nonlinear; video-stabilization work often smooths them in a better-behaved parameterization such as a log-homography model rather than averaging raw matrices. ([CVF Open Access][2])

So I would say:

## Your idea is good as **cross-frame consistency / temporal fusion**

## but bad as **raw matrix averaging**

The cleaner version is:

### 1. Treat the direct prediction as one candidate

For source `s` and target `t`, you already have

[
\hat H_{s\to t}^{direct}
]

### 2. Use intermediate frames as extra candidates

For each intermediate frame `i`, predict

[
\hat H_{s\to i}, \quad \hat H_{i\to t}
]

and compose them:

[
\hat H_{s\to t}^{(i)} = \hat H_{i\to t}\hat H_{s\to i}
]

This part is fully sensible. Composition is exactly the right operation for chaining homographies. ([UW Computer Sciences][3])

### 3. Do **consistency filtering**, not blind averaging

Now you have a set of candidate transforms:

* direct
* via frame (i_1)
* via frame (i_2)
* via frame (i_3), etc.

Instead of averaging matrices directly, do one of these:

#### Option A: choose the best candidate

Evaluate each candidate by warping the source ROI and scoring alignment against the target with your masked loss.
Pick the one with the lowest score.

This is the safest first version.

#### Option B: robustly fuse in parameter space

Convert each candidate homography into a local parameterization around identity, then average those parameters robustly.

For example, parameterize the residual transform as:

* 4 corner offsets, or
* translation + rotation + scale + shear + small perspective terms

Then take a weighted robust mean in that parameter space.

This is much safer than averaging 3x3 entries.

#### Option C: confidence-gated correction

Use the direct prediction unless the multi-frame candidates strongly agree on a different answer.

This avoids ruining already good alignments.

That last point matters because you already observed the current failure mode:

> sometimes it adds jitter/drift to an already well-aligned ROI

That usually means the model is **over-correcting noisy micro-errors**. So any fusion rule should be conservative.

---

# What is probably happening now

Your Stage 1 visuals look great because the network learned the synthetic task well.

But in the actual pipeline:

* the source-target pair is not identical
* real tracks contain blur, cropping inconsistency, illumination shift, slight OCR box changes, maybe partial occlusion
* the network predicts a residual correction even when the correct answer is “almost do nothing”

That creates:

* drift
* frame-to-frame wobble
* overfitting to weak local cues

So the real issue is likely not “the network is useless,” but:

> it lacks a strong mechanism to decide when **not** to correct.

Your cross-reference idea is one way to provide that missing stability.

---

# Best reformulation of your idea

I would frame it as **cycle-consistency / multi-hop consistency on the track**.

For a stable track, good predictions should satisfy approximately:

[
H_{s\to t} \approx H_{i\to t}H_{s\to i}
]

and more generally:

[
H_{s\to t} \approx H_{j\to t}H_{i\to j}H_{s\to i}
]

So instead of saying “average lots of transforms,” say:

> use intermediate frames to validate or refine the direct transform through path consistency.

That sounds more principled, and it is.

---

# What I would implement first

## Version 1: candidate selection by masked alignment score

For each source-target pair:

1. predict direct `H_direct`
2. sample a few nearby intermediate frames `i`
3. predict `H_si` and `H_it`
4. compose `H_via_i = H_it * H_si`
5. evaluate each candidate on the target using your masked score:

   * masked gradient loss
   * masked SSIM / robust photometric term
6. choose the best-scoring candidate
7. only apply it if it is better than `H_direct` by a margin

This is simple and conservative.

Why I like it:

* no fragile homography averaging
* easy to debug
* directly tied to actual alignment quality
* naturally avoids changing a good direct result unless there is evidence

## Version 2: robust median / mean in corner space

If you really want fusion instead of selection:

1. convert each candidate homography into warped positions of the 4 source patch corners in target space
2. now each candidate becomes 8 numbers
3. do a robust aggregation:

   * weighted median per coordinate, or
   * Huber-weighted mean after outlier rejection
4. reconstruct final homography from the aggregated corners

This is much more reasonable than averaging matrices.

Because in your setup the residual transform is small, corner-space averaging is often good enough.

---

# Very important guardrail

You should add a **do-no-harm gate**.

For example:

* compute the masked alignment score under `H0` alone
* compute the score under `H_refined`
* only use the refined transform if it improves the score by at least some threshold

Otherwise keep `H0`.

This may help more than any multi-frame fusion.

Because your actual complaint is not just low accuracy. It is:

> sometimes refinement makes good cases worse

That is exactly what a conservative gate is for.

---

# Better than post-hoc fusion: train with consistency

If you have time, the more principled version is to add a **consistency loss** during training:

For triplets `(s, i, t)` from the same track, encourage

[
H_{s\to t} \approx H_{i\to t}H_{s\to i}
]

in corner space or parameter space.

This would directly teach the network that trackwise transforms should compose consistently.

But with your deadline, I would not retrain heavily unless integration is otherwise hopeless.

For now, post-hoc consistency filtering is more realistic.

---

# What not to do

Do not:

* average raw 3x3 matrices directly
* average matrices without scale normalization
* trust all intermediate frames equally
* use distant frames with very different appearance or weak overlap
* always replace the direct result with the fused result

All of those can add more jitter rather than remove it.

---

# Best practical recipe under deadline

If I were optimizing for “works before deadline,” I would do this:

### For each pair `(s,t)`:

* direct candidate: `H_direct`
* pick 2–4 nearby intermediate frames only
* build `H_via_i = H_it * H_si`
* score all candidates using the same masked image-space criterion
* reject outlier candidates
* choose the best one
* apply only if it beats the current alignment by a margin
* otherwise keep coarse alignment or direct refinement

That gives you:

* cross-reference behavior
* no dangerous matrix averaging
* explicit protection against added jitter

---

# Bottom line

Your idea **does make sense**, but the right concept is:

**multi-frame consistency and candidate fusion**,
not
**simple weighted averaging of homography matrices**.

The strongest immediate version is:

* compose through intermediate frames,
* evaluate each candidate on the target,
* pick or robustly fuse in **corner space**,
* and use a **do-no-harm gate** so good alignments stay good.

That is much more likely to reduce jitter than raw averaging.

If you want, I can next give you a concrete pseudocode design for:

1. candidate generation,
2. scoring,
3. corner-space fusion, and
4. the do-no-harm gate.

[1]: https://www.researchgate.net/publication/224218374_Closed-Form_Solutions_to_Multiple-View_Homography_Estimation?utm_source=chatgpt.com "Closed-Form Solutions to Multiple-View Homography ..."
[2]: https://openaccess.thecvf.com/content/WACV2021/papers/Bradley_Cinematic-L1_Video_Stabilization_With_a_Log-Homography_Model_WACV_2021_paper.pdf?utm_source=chatgpt.com "Cinematic-L1 Video Stabilization With a Log-Homography ..."
[3]: https://pages.cs.wisc.edu/~dyer/cs534/papers/szeliski-alignment-tutorial.pdf?utm_source=chatgpt.com "Image Alignment and Stitching: A Tutorial"


# [Me]:

I think I have time to train some more epochs. How do I add a consistency loss? They network only takes a pair of ROIs as input and each training iteration sames two ROIs from the same track in the current implementation.


# [ChatGPT]:

Yes. You can add a consistency loss **without changing the model input format**.
The network can still take only **one pair of ROIs** each forward pass. The change is in how you build a minibatch item and how you combine **multiple forward passes** into one loss.

The right way is to switch from a **pair sample** to a **triplet sample** during training:

* source frame `s`
* intermediate frame `i`
* target frame `t`

Then run the same network **three times**:

* predict `H_si` from `(s, i)`
* predict `H_it` from `(i, t)`
* predict `H_st` from `(s, t)`

and add a loss that encourages

[
H_{st} \approx H_{it} H_{si}
]

That is the standard kind of cycle / compositional consistency used in geometric learning: the transform predicted for a long hop should agree with the composition of shorter hops. Unsupervised homography work commonly combines reprojection-style objectives with cycle-consistency ideas, and recent surveys of deep homography estimation explicitly group cycle-consistency with photometric reprojection as a standard unsupervised objective family. ([arXiv][1])

## What to change in data loading

Right now each training iteration samples two ROIs from the same track.
Change that to sample **three** ROIs from the same track:

* choose a track with length at least 3
* sample ordered frame indices `s < i < t`
* keep them reasonably close at first

Good default:

* `i = s + 1`
* `t = i + 1`

Then later allow wider gaps, like:

* `i = s + rand(1, 2)`
* `t = i + rand(1, 2)`

Do not start with large gaps, because consistency becomes noisy when appearance changes too much.

## What to predict

Use the exact same network output as before.

If your network outputs:

* 4-corner offsets, or
* residual homography `ΔH`

that is fine.

But for the consistency loss, do **not** compare raw 3x3 matrix entries directly unless you are very careful with normalization, because homographies are defined only up to scale and the space is nonlinear. This is exactly why homography-related work often uses better-behaved parameterizations or normalized forms rather than naïve matrix arithmetic. ([arXiv][1])

## Best practical consistency loss

### Preferred version: corner-space consistency

For a canonical source patch with 4 corners:

[
C = {(0,0),(W,0),(W,H),(0,H)}
]

compute:

* `C_direct = warp_corners(H_st, C)`
* `C_via = warp_corners(H_it * H_si, C)`

Then define:

[
L_{cons} = \text{smoothL1}(C_{direct}, C_{via})
]

This is the safest implementation.

Why this is good:

* easy to code
* scale ambiguity is avoided
* directly tied to geometric effect
* robust when residuals are small

For your setup, this is better than matrix-entry loss.

## Alternative: parameter-space consistency

If your output is 8 corner displacements, you can convert both:

* direct prediction `H_st`
* composed prediction `H_it * H_si`

into 4 warped corners in target coordinates, then compare those 8 values.

That is essentially the same as above.

## Full loss structure

You probably already have something like:

[
L = L_{pair}
]

where `L_pair` is your existing supervised or self-supervised pair loss.

Now change it to:

[
L = L_{si} + L_{it} + L_{st} + \lambda_{cons} L_{cons}
]

where:

* `L_si` = existing loss on pair `(s,i)`
* `L_it` = existing loss on pair `(i,t)`
* `L_st` = existing loss on pair `(s,t)`
* `L_cons` = consistency loss comparing `H_st` with `H_it H_si`

If you are already in Stage 2 with masked alignment losses, each `L_ab` can remain exactly your current masked loss.

## If you already have coarse alignment `H0`

Even better: do consistency on the **final predicted transform**, not just the residual.

If your network predicts residuals:

[
H_{ab} = \Delta H_{ab} H^0_{ab}
]

then use those final transforms in the consistency relation.

Compare:

[
H_{st}
\quad \text{vs} \quad
H_{it} H_{si}
]

in corner space.

That keeps the consistency loss aligned with what is actually used at inference.

## Very important practical warning

In real tracks, exact composition will **not** hold perfectly because:

* crops are not perfectly canonical,
* OCR boxes jitter,
* motion blur changes local evidence,
* surfaces may not be perfectly planar,
* your coarse alignment may inject noise.

So the consistency loss should be **soft**, not dominant.

Start with a small weight like:

* `λ_cons = 0.05`
* or `0.1`

and only increase if it clearly helps.

If you make it too strong, the network may become overly smooth and stop correcting useful local errors.

## Even lighter version if you are short on time

You do not need a full triplet loss on every batch item.

A simpler implementation is:

* keep normal pair training for most samples
* for a fraction of batch items, sample triplets and add consistency

For example:

* 70% ordinary pairs
* 30% triplets with consistency

That reduces implementation disruption.

## Best starter formulation for your codebase

Assume your current training step looks like:

```python
pred = model(src, tgt)
loss = pair_loss(pred, src, tgt, ...)
```

Change it to:

```python
pred_si = model(img_s, img_i)
pred_it = model(img_i, img_t)
pred_st = model(img_s, img_t)

H_si = compose(pred_si, H0_si)   # or just pred_si if no H0
H_it = compose(pred_it, H0_it)
H_st = compose(pred_st, H0_st)

loss_pair = loss_si + loss_it + loss_st

corners = canonical_corners(W, H)
corners_direct = warp_corners(H_st, corners)
corners_via = warp_corners(H_it @ H_si, corners)

loss_cons = smooth_l1(corners_direct, corners_via)

loss = loss_pair + lambda_cons * loss_cons
```

That is the core idea.

## Strong recommendation: add a confidence gate

Because you already observed drift/jitter, I would not enforce consistency equally for all triplets.

Use a simple reliability weight, for example based on:

* pair reconstruction score
* mask overlap
* temporal distance

Then:

[
L_{cons}^{weighted} = w_{triplet} \cdot L_{cons}
]

Good first choice:

[
w_{triplet} = \exp(-\alpha (d_{si}+d_{it}+d_{st}))
]

where `d_ab` is your masked alignment error for pair `(a,b)`.

Interpretation:

* if the pair predictions already look unreliable, trust the consistency term less

## Another good option: only enforce local consistency

Since your model seems decent but unstable, start with only adjacent triplets:

* `(k, k+1, k+2)`

This is much less noisy than random wide triplets.

Once that works, widen the temporal span a bit.

## What to watch out for

The main failure modes are:

1. **Too strong consistency weight**
   The model stops making useful corrections and collapses toward over-smoothed transforms.

2. **Comparing raw homography matrices**
   This is fragile and can behave badly because of scale ambiguity. Use corner-space comparison instead. ([arXiv][1])

3. **Long-gap triplets too early**
   Appearance differences and planarity violations make the consistency target noisy.

4. **Wrong composition order**
   Check carefully whether your convention is:

   * `H_it @ H_si`
     or
   * `H_si @ H_it`

   This depends on whether your homographies map source-to-target or target-to-source.

5. **No baseline pair loss**
   Consistency alone is not enough. It should regularize pair training, not replace it.

## My recommendation for your deadline situation

Do this minimal version:

* sample adjacent triplets from the same track
* run the same pairwise network 3 times
* compute existing pair losses as usual
* add a small corner-space consistency loss between direct `s→t` and composed `s→i→t`
* set `λ_cons = 0.05` to start
* train a few epochs
* compare temporal jitter visually

That is the highest-value low-risk modification.

If you want, I can write the exact tensor-level pseudocode next, assuming your network outputs either 8 corner offsets or a 3x3 homography.

[1]: https://arxiv.org/pdf/2601.18222?utm_source=chatgpt.com "HomoFM: Deep Homography Estimation with Flow Matching"
