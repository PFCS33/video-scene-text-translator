# Plan: Web Client UI Redesign — slate-dark, two-column app shell

## Goal
Swap the web client's centered-card MVP UI for the slate-dark, two-column layout specced in `web/mockup-handoff/`. Frontend-only — no server API changes. Cover six visual states (Idle, Uploading, Connecting, Running, Succeeded, Failed, Rejoin) with real upload progress, live stage telemetry, and the log-panel treatment from the mockup.

## Branch
`feat/web-client` — same branch as the MVP. The redesign extends it.

## Approach

### Decisions captured (from the prior discussion)

- **D1 — Upload progress via XHR.** Replace `fetch` in `createJob` with `XMLHttpRequest` so the Uploading state can show real `%` / MB per second / ETA. New `onProgress` callback on `createJob`. `fetch`-based helpers for non-upload endpoints stay.
- **D2 — shadcn where close, custom where mockup has its own vocabulary.** Keep `<Button>`, `<Alert>`, `<Card>`, `<Badge>` re-skinned via tokens. Hand-roll the mockup-specific primitives: `.lang-select`, stage tile, job chip, log line. Matches web/CLAUDE.md D6.
- **D3 — Minimal video preview.** Native `<video controls>` + `URL.createObjectURL(file)` + filename + size. No custom scrubber, no fps/resolution probe.
- **D4 — Test rewrites ship with each component.** A component change and its `__tests__/` rewrite land in the same commit. `npm run test` stays green commit-by-commit. Expect the final test count to differ from today's 60 — that's fine.
- **D5 — Keep technical stage names.** Detect / Frontalize / Edit / Propagate / Revert. Aligns with backend `s1..s5` codes and the mockup.
- **D6 — Drop the macOS window chrome.** Traffic lights, URL bar, top-bar job chip → mockup-only flourish. Production keeps the fixed-width two-column app shell only. Status-at-a-glance info that the chrome carried (job id + phase) moves into a small status band above the right column or onto the right-column pill — whichever reads cleaner in practice.
- **D7 — Collapse the `<UploadForm>` / `<JobView>` split.** The mockup's left column is identical across all six states (only badge/lock state and submit-button variant change). `<App>` becomes the state-machine owner; `<AppShell>` is a presentational two-column frame; the left column is one stateless composite component driven entirely by props. The right column is a switch over a discriminated `uiState`. `UploadForm` and `JobView` files are deleted.
- **D8 — Reuse `useJobStream` with one additive change.** Extend the hook to accept an explicit `phase` seed (`uploading | connecting | running | …`) and emit an active-stage elapsed tick (a `useEffect` `setInterval` bound to the current `stage_start.ts`). No change to event-folding semantics.
- **D9 — Rejoin fetches blocking job's `/status` on 409.** When `createJob` returns 409 with `active_job_id`, App calls `getJobStatus(active_job_id)` to populate the RejoinCard's metadata (`current_stage`, `created_at`). If the fetch fails, the card still renders with id + generic copy. "from other session" marker and the heuristic ETA are omitted.
- **D10 — No backend changes.** Every step in this plan is local to `web/`. If we discover a real gap during implementation we stop and revisit.

### UI state machine (owned by `<App>`)

```ts
type UiState =
  | { phase: "idle"; file: File | null; source: string; target: string; submitError?: SubmitError }
  | { phase: "uploading"; file: File; source: string; target: string; progress: UploadProgress }
  | { phase: "rejoin"; file: File; source: string; target: string; blockingJobId: string; blockingStatus?: JobStatus }
  | { phase: "active"; jobId: string; file: File | null; source: string; target: string }
```

The `active` phase delegates to `useJobStream` for the finer `connecting | running | succeeded | failed` slice. Transitions:
- `idle` → `uploading` on submit (XHR starts).
- `uploading` → `active` on XHR load (201 + `job_id`). File + langs preserved for the left column's locked video card.
- `uploading` → `rejoin` on XHR 409. File + langs preserved; a `/status` fetch populates blocking-job metadata.
- `uploading` → `idle` on XHR error (re-render with a `SubmitError`).
- `rejoin` → `active(jobId = blockingJobId, file = null)` on "Rejoin running job" click.
- `active` (terminal) → `idle` on "Submit another" or "Delete job" (clears file).

### Design tokens + theme

- Copy `web/mockup-handoff/design/tokens.css` into `web/src/styles/globals.css`'s `:root` block, replacing the shadcn slate tokens. Then map shadcn's semantic vars onto the new palette so primitives inherit correctly:
  - `--background` ← `--bg-0`, `--foreground` ← `--ink-0`
  - `--card` ← `--bg-1`, `--card-foreground` ← `--ink-0`
  - `--popover` ← `--bg-2`
  - `--primary` ← `--acc`, `--primary-foreground` ← `--bg-0`
  - `--secondary` ← `--bg-3`
  - `--muted` ← `--bg-2`, `--muted-foreground` ← `--ink-2`
  - `--accent` ← `--bg-3`
  - `--destructive` ← `--err`, `--destructive-foreground` ← `--ink-0`
  - `--border` ← `--line`, `--input` ← `--line-2`, `--ring` ← `--acc-line`
- Delete the `.dark` block — we ship a single dark theme. `--radius: 0.5rem` stays.
- `web/tailwind.config.ts`: add `fontFamily.sans` → `var(--ff-sans)`, `fontFamily.mono` → `var(--ff-mono)`. Keep the existing shadcn-derived `extend.colors` shims so `bg-background`, `text-muted-foreground`, etc., still work.
- `web/index.html`: add Inter + JetBrains Mono via Google Fonts `<link>` (`preconnect` + `css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap`).

### App shell

- Fixed width 1080 px, height 760 px, centered on the page (`min-h-screen flex items-center justify-center`).
- Two columns inside: left `400px` with `border-r border-line bg-[--bg-1]`, right `flex-1 bg-[--bg-0]`.
- No titlebar / traffic lights / URL bar (D6).
- Below 1080 viewport width → render a "Desktop required · at least 1080px wide" card (one breakpoint, one component).

### Right-column surfaces (one per phase)

| Phase | Right-column component |
|---|---|
| idle | `<IdlePlaceholder>` — centered glyph + "WAITING FOR A JOB" + copy |
| uploading | `<UploadProgress>` — big %, `N / M MB · R MB/s`, bar, `~S remaining` |
| connecting | `<StageProgress>` with all pending + elapsed 0 (no log yet) |
| running | `<StageProgress>` + elapsed row + `<LogPanel>` |
| succeeded | `<StageProgress>` (all done + totals row) + `<ResultPanel>` |
| failed | `<StageProgress>` (failing stage red, later pending) + `<FailureCard>` |
| rejoin | `<RejoinCard>` — blocking-job metadata + Rejoin CTA |

### Left-column variants (one composite, props-driven)

`<LeftColumn>` takes `{ fileSlot, langLocked, submitSlot, statusFooter? }` and always renders the same skeleton:
```
IDENTITY → Scene Text Translator + tagline
INPUT    → <fileSlot />   ← Dropzone | VideoCard variant
LANGUAGES→ <LanguagePair locked=langLocked />
flex-spacer
<submitSlot />            ← SubmitBar variant
```
The five file-slot / submit-slot variants (idle, uploading, rejoin, running/succeeded/failed) are small wrappers that set badge + label + disabled state. No per-phase layout divergence.

### MVP keepers (cheap, in scope)

- **Language swap ↕** — one click swaps `source` ↔ `target`.
- **Copy-error button** on FailureCard — `navigator.clipboard.writeText(message + traceback)`.
- **⌘↵ submit keybind** — document-level `keydown` listener in `<App>` guarded on `phase === "idle"`.
- **`prefers-reduced-motion`** — CSS media query in globals.css kills the stripe-shimmer and dot-pulse animations.
- **Desktop-required card** for < 1080 px.
- **Active-stage running tick** — one `setInterval(1s)` inside `useJobStream`, cleared on stage change or unmount. Feeds the active stage tile's "32s" readout and the status band's "S3/5 · 00:42".
- **Bold `═══ Stage N ═══` log separator** — regex match on log message in the renderer, `.hdr` class.
- **Static "silence is OK. Stage 3 can pause…" hint** — always rendered as a `.dim` line at the tail of the log when `currentStage === "s3"` and phase is running.

### Deferred (mockup-introduced; backend doesn't support or scope creep)

- ETA heuristic (`~1:10`) on Running + Rejoin — omit the line entirely.
- "4 regions rewritten" summary stat on Succeeded — omit.
- Rejoin "from other session" marker — omit.
- Rejoin "your file stays queued" auto-resubmit — keep the copy, skip the behavior.
- `↻ replace` chip on the video card during upload (abort XHR + swap) — defer.
- Retry button on Failed — defer; "Submit another" covers it.
- Human-friendly error title mapping — render raw `error.message` as title, full traceback in `<details>`.
- localStorage page-reload rejoin — MVP rejoin = 409-on-submit only.
- "Jump to latest" chip when log is scrolled up — plain auto-scroll for MVP.

## Files to Change

### Tokens + theme
- [ ] `web/src/styles/globals.css` — replace `:root` tokens with a verbatim copy of `mockup-handoff/design/tokens.css`, add the shadcn-var → slate-token mapping; delete `.dark` block; add `@media (prefers-reduced-motion: reduce) { ... }` stopping `--tw-animate-*` on stripe/pulse utilities.
- [ ] `web/tailwind.config.ts` — add `fontFamily.sans` / `fontFamily.mono`; keep existing colors extension.
- [ ] `web/index.html` — preconnect + Inter + JetBrains Mono `<link>`.

### API client
- [ ] `web/src/api/client.ts` — rewrite `createJob` on XHR: params `(file, source, target, { onProgress?, signal? })`; return `Promise<JobCreateResponse>`; emit `{ loaded, total, elapsedMs }` snapshots via `onProgress`. `getHealth/getLanguages/getJobStatus/deleteJob` stay on `fetch`. Keep `ApiError`/`ConcurrentJobErrorDetail` + `outputUrl/eventsUrl` helpers unchanged.
- [ ] `web/src/api/schemas.ts` — add `UploadProgress` type (`loaded, total, percent, bytesPerSec, etaSeconds`).
- [ ] `web/src/api/__tests__/client.test.ts` — add XHR progress tests with `vitest`'s fake XHR (or a hand-rolled mock); keep all non-upload tests.

### Hooks
- [ ] `web/src/hooks/useJobStream.ts` — add `activeStageElapsedMs` to state, driven by a `setInterval` bound to the latest `stage_start.ts`; interval cleared on stage change + unmount + terminal. No change to event-folding.
- [ ] `web/src/hooks/__tests__/useJobStream.test.ts` — add tick test (fake timers), regression-check existing assertions.

### App shell + layout
- [ ] (new) `web/src/components/AppShell.tsx` — fixed-size 1080×760 two-column frame + desktop-required fallback card.
- [ ] (new) `web/src/components/DesktopRequired.tsx` — simple card shown below 1080 px.
- [ ] `web/src/App.tsx` — state-machine owner: `useState<UiState>`; handles XHR submit + 409 → rejoin, rejoin → active, active → idle; renders `<AppShell>` + left/right slots. Wires ⌘↵ keybind.

### Left column
- [ ] (new) `web/src/components/left/LeftColumn.tsx` — composite, stateless, props-driven (file slot, language pair, submit slot, optional footer).
- [ ] (new) `web/src/components/left/IdentityBlock.tsx` — title + tagline.
- [ ] `web/src/components/Dropzone.tsx` — re-skinned to match mockup `.drop` (dashed card, upload glyph, "Drop video here", "or click to pick", constraints tiny line). Drops oversize inline-red text under card.
- [ ] (new) `web/src/components/left/VideoCard.tsx` — `<video controls src=blobUrl>` + corner tag (INPUT / INPUT · EN / YOUR INPUT · QUEUED) + filename + size. Variant prop selects which tag + footer text.
- [ ] `web/src/components/LanguageSelect.tsx` — hand-rolled flat select matching `.lang-select` look (caret, truncate). Controlled, disabled prop, optional `locked` visual.
- [ ] (new) `web/src/components/left/LanguagePair.tsx` — pair of `<LanguageSelect>` + swap button `↕` between them. Disabled prop. Mono footer line (`● LOCKED WHILE RUNNING` etc.).
- [ ] (new) `web/src/components/left/SubmitBar.tsx` — primary button + hint row + optional `✗ delete job` link. Variant prop drives label + style (`disabled` / `uploading` / `running` / `submit-another` / `waiting`).

### Right column
- [ ] (new) `web/src/components/right/StatusBand.tsx` — thin label row (e.g. `PIPELINE · ONE WINDOW`) + status pill (`IDLE` / `CLIENT → SERVER` / `● LIVE` / `READY` / `● ERR` / `● BLOCKED`).
- [ ] (new) `web/src/components/right/IdlePlaceholder.tsx` — centered glyph + headline + copy.
- [ ] (new) `web/src/components/right/UploadProgress.tsx` — big %, `loaded / total · MB/s`, thin bar, `~S remaining`, copy.
- [ ] `web/src/components/StageProgress.tsx` — rewrite to 5 stage tiles with `S1..S5` number + name + duration/tick, state classes `done | active | pending | fail`. Elapsed row with clock glyph + stripe meter.
- [ ] `web/src/components/LogPanel.tsx` — rewrite: `.log-head` + scrollable `.log` list; timestamp · severity chip · body; regex-bold `═══ Stage N ═══` lines; static S3 silence hint; plain auto-scroll.
- [ ] `web/src/components/ResultPanel.tsx` — rewrite: `<video controls src=outputUrl>` + OUTPUT corner tag + success-styled Download button.
- [ ] (new) `web/src/components/right/FailureCard.tsx` — replaces `ErrorAlert`. Red label + raw `error.message` title + one-line description (fallback to first traceback line) + `<details>` traceback + Retry (visually present but disabled for MVP? — no, omit) → **just Copy error + no Retry button for MVP**. Copy error wires to `navigator.clipboard`.
- [ ] `web/src/components/ErrorAlert.tsx` — **delete**; replaced by FailureCard.
- [ ] (new) `web/src/components/right/RejoinCard.tsx` — yellow label + "Server is busy" + metadata list (`running job`, `stage`, `started`) + full-width Rejoin primary button + footer copy.

### Deletions
- [ ] `web/src/components/UploadForm.tsx` — delete (merged into `<App>` + `<LeftColumn>` + right-column surfaces).
- [ ] `web/src/components/JobView.tsx` — delete (merged likewise).

### Tests
- [ ] `web/src/components/__tests__/*` — rewrite in step with each component change. New tests: `AppShell`, `VideoCard`, `UploadProgress`, `RejoinCard`, `FailureCard`, `LanguagePair` (swap behavior), state-machine transitions at the `<App>` level.

## Risks

- **R1 — State-machine rewrite touches every post-MVP component.** Collapsing `UploadForm` + `JobView` into an App-level reducer is the single largest diff. Mitigation: build the shell + primitives first, port screens one at a time behind a temporary `phase` switch while the old `UploadForm`/`JobView` still exist, then flip the switch in the final step. Keep `npm run test` green throughout.
- **R2 — XHR progress events are browser-flaky.** Some browsers coalesce progress events; a 200 MB upload might fire only a few `progress` events on fast LANs. The UI must degrade to "known bytes sent / total" without divide-by-zero MB/s math. Mitigation: guard `bytesPerSec` by elapsed ≥ 1s and show `—` until we have one data point.
- **R3 — shadcn re-skin drift.** After mapping shadcn semantic vars onto slate tokens, primitives like `<Button variant="destructive">` may look off (e.g. destructive bg → `--err` which is a light red, fine against dark, but the destructive-foreground contrast needs a sanity check). Mitigation: manually verify each of the four shadcn primitives (Button/Alert/Card/Badge) in all their variants after the theme swap.
- **R4 — `<video>` blob URL memory leak.** `URL.createObjectURL(file)` leaks unless `URL.revokeObjectURL` runs. Mitigation: `useEffect` cleanup in `<VideoCard>`.
- **R5 — Test suite regressions.** Many current tests assert on `CardTitle`/`Submit another` etc. which stop existing mid-migration. Mitigation: per-component commit rhythm (D4); never let main break.
- **R6 — Live SSE + XHR upload running simultaneously is not a thing.** The app never does both at once (upload finishes before SSE opens), so there's no resource contention. Just noting it explicitly so no one writes a test trying to do both.
- **R7 — Regex on log messages for stage-separator bolding is fragile.** If the pipeline's log format drifts, the bolding silently stops matching. Mitigation: match a loose pattern (`/^={3,}\s*Stage\s+\d/i`) and accept that a rename server-side will need a client-side regex update — noted as a follow-up.
- **R8 — ⌘↵ keybind conflicts.** Some browsers intercept Cmd+Enter in contenteditable; we have none, so safe. Still, guard the listener to only fire when `phase === "idle"` + no modal/focused input traps it.
- **R9 — Reduced-motion users still need phase feedback.** If we nuke the stripe + pulse, the only remaining signal on Running is the stage-tile state flip. Mitigation: swap the pulse for a static accent ring on the active tile under reduced-motion — no animation, but still visually distinct.
- **R10 — Delete button position changes UX.** In the mockup, delete is a ghost `✗ delete job` link at the bottom-right of the left column's submit hint. Dangerously close to "Submit another". Mitigation: keep the current one-click behavior for MVP (brief §12 Q8 flags adding confirmation as a later question) but style it `text-muted-foreground` so it doesn't read as primary.

## Done When

- [ ] All six states render pixel-close to `mockup-handoff/screenshots/0{1..6}-*.png` at 1080×760 in the dev browser.
- [ ] A real dev-server upload shows live `%`/`MB per s`/ETA during Uploading and transitions cleanly to Running on server 201.
- [ ] 409 on submit lands on the RejoinCard with the blocking job's current stage + start time fetched from `/status`; clicking Rejoin switches to the running view of the blocking job.
- [ ] End-to-end flow (upload → 5-stage SSE → download) works against the real backend and the output MP4 plays in the Succeeded-state video element.
- [ ] Pipeline Failed path shows the FailureCard with raw message + collapsible traceback + Copy error (clipboard write verified).
- [ ] < 1080 px viewport shows the Desktop-required card instead of the app.
- [ ] `prefers-reduced-motion: reduce` kills the stripe and pulse animations; active stage is still visually distinct.
- [ ] `npm run test` green; `npm run type-check` clean; `npm run lint` clean.
- [ ] `ruff check server/` still clean (no server changes expected, but re-verify nothing slipped).
- [ ] `web/CLAUDE.md` updated with the new component layout + any new gotchas.
- [ ] `docs/architecture.md` "Web Application" section updated to reflect the state machine + collapsed UploadForm/JobView.
- [ ] Code review by `@reviewer` — feedback addressed.
- [ ] Changes committed as atomic commits on `feat/web-client`.

## Progress

- [x] **Step 1 — Design tokens + fonts + theme swap.** Port `tokens.css` verbatim into `globals.css`; map shadcn semantic vars onto slate tokens; delete `.dark` block. Add fonts to `index.html`. Extend Tailwind config with font families. **Goal: the existing UploadForm/JobView still renders, but in the new slate-dark palette.** No layout changes yet. Verify all four shadcn primitive variants read correctly (R3).
  - Trap caught: shadcn's Tailwind config wrapped color vars as `hsl(var(--foo))`, but slate tokens are hex / rgba. Fix: drop the `hsl(...)` wrapper in `tailwind.config.ts`.
  - `body { font-family: var(--ff-sans); }` added so Inter applies globally without opt-in.
  - `prefers-reduced-motion` block placed outside `@layer base` (plain CSS) so `!important` beats Tailwind animation utilities.
  - R3 still open — no display on dev box; four shadcn primitive variants (Button / Alert / Card / Badge) type-check + test-green but need human visual sign-off next session, especially `destructive-foreground` contrast on `--err` and `ring` visibility at 35% opacity.
  - `npm run type-check` / `npm run lint` / `npm run test` (60/60) all green.
- [ ] **Step 2 — `createJob` on XHR with `onProgress`.** Rewrite to XMLHttpRequest; add `UploadProgress` type to `schemas.ts`; add `AbortSignal` support. Tests with a mock XHR. No consumer wiring yet.
- [ ] **Step 3 — `useJobStream` active-stage tick.** Add `activeStageElapsedMs` driven by `setInterval(1000)`. Update existing tests with fake timers. No consumer UI change.
- [ ] **Step 4 — `<AppShell>` + `<DesktopRequired>`.** Build the fixed 1080×760 two-column frame with slot props for left + right. Viewport breakpoint card. Tests: it renders slot children; it shows the desktop-required card below 1080.
- [ ] **Step 5 — Dropzone + `<VideoCard>`.** Reskin Dropzone; add VideoCard with blob URL + cleanup. Component tests: file select, oversize warn, video cleanup on unmount (R4).
- [ ] **Step 6 — `<LanguageSelect>` + `<LanguagePair>` (with swap).** Hand-rolled flat select; pair composite with swap button + mono footer. Tests: controlled behavior, swap, locked mode.
- [ ] **Step 7 — `<SubmitBar>` + `<IdentityBlock>` + `<LeftColumn>`.** Five submit variants; left-column composite. Tests: variant labels + disabled states.
- [ ] **Step 8 — Right-column idle + upload surfaces.** `<IdlePlaceholder>`, `<UploadProgress>`, `<StatusBand>`. Tests: renders copy; UploadProgress math (degrades when MB/s unknown — R2).
- [ ] **Step 9 — `<StageProgress>` rewrite.** 5 tiles + elapsed row + stripe meter (reduced-motion safe — R9). Tests: state classes per tile; active-tile tick readout.
- [ ] **Step 10 — `<LogPanel>` rewrite.** Mono list, severity chip, bold stage separators (regex — R7), static S3 hint, plain auto-scroll. Tests: separator regex, severity rendering, auto-scroll behavior.
- [ ] **Step 11 — `<ResultPanel>` rewrite.** Output video + corner tag + success-styled download. Tests: renders output URL, download link.
- [ ] **Step 12 — `<FailureCard>` + delete `<ErrorAlert>`.** Copy-error via clipboard; collapsible traceback. Tests: renders message + traceback; copy click writes to clipboard.
- [ ] **Step 13 — `<RejoinCard>` + 409 `/status` fetch.** Fetch `getJobStatus(activeJobId)` on 409; render id + current stage + started-at; Rejoin CTA. Tests: renders with + without blocking-status fetch; Rejoin click invokes callback.
- [ ] **Step 14 — App state machine + delete UploadForm/JobView.** Collapse into `<App>` reducer owning `UiState`; wire XHR submit, 409 handling, terminal transitions. ⌘↵ keybind. Delete `UploadForm.tsx` and `JobView.tsx`. App-level integration tests for phase transitions.
- [ ] **Step 15 — Cosmetic polish pass.** Mono footers ("● LOCKED WHILE RUNNING" etc.), active-tile accent, hint-row delete-link styling (R10), prefers-reduced-motion coverage. One round of visual diff against screenshots.
- [ ] **Step 16 — Dev-box smoke.** Real upload + real pipeline + real download against the backend. Verify against all six state screenshots. Log any drift in a follow-up.
- [ ] **Step 17 — Update `web/CLAUDE.md` + `docs/architecture.md`.** Reflect the collapsed UploadForm/JobView, App-level state machine, D6–D10 decisions.
- [ ] **Step 18 — `@reviewer` pass.** Address feedback.
- [ ] **Step 19 — Atomic commits.** Each Step above is one commit (or a tight sequence if a step naturally splits). Conventional format `feat(web)` / `chore(web)` / `test(web)`.
