# web/ — React frontend for the live demo

Nested CLAUDE.md — auto-loads when working on the React SPA. Root CLAUDE.md
covers the pipeline; this file only covers the frontend.

## Scope
React 18 + Vite + TypeScript SPA that drives the FastAPI server in
`server/`. Fixed 1080×760 two-column `<AppShell>` driven by a `UiState`
state machine in `App.tsx` — four phases (`idle | uploading | rejoin |
active`). Upload video → pick languages → watch per-stage progress + live
log via SSE → download the output MP4.

## Run
```bash
source /opt/nvm/nvm.sh              # Node >= 20 via nvm
cd web
npm install                         # or npm ci for reproducible installs
npm run dev                         # Vite :5173, proxies /api to :8000
npm run test                        # vitest + jsdom (129 tests)
npm run build                       # tsc -b && vite build → dist/
npm run type-check                  # tsc --noEmit
```

## Component-reuse principle (D6) — in priority order
1. **shadcn primitives first.** Use `<Button>`, `<Alert>`, `<Card>`,
   `<Badge>`, `<Input>`, `<Label>` from `components/ui/*` as-is
   (re-skinned via slate tokens in `globals.css`). Add more with
   `npx shadcn@latest add <name>` (config lives in `components.json`).
2. **Wrap-and-extend second.** If a primitive needs a thin domain shim,
   keep the wrapper stateless: controlled props in, shadcn primitive out.
   No hidden state, no style overrides.
3. **Custom from scratch last.** The mockup has its own vocabulary
   (`.lang-select`, stage tile, log line, stripe meter) that shadcn
   doesn't cover — hand-roll these. Tier-3 custom components today:
   `Dropzone`, `StageProgress` (5 discrete tiles), `LogPanel`,
   `LanguageSelect` (native `<select>`, Radix was dropped), plus the
   phase-specific surfaces under `components/left/` and
   `components/right/`. Bind Tailwind classes to the design tokens in
   `src/styles/globals.css` so custom components stay coherent.

Before adding a custom component, ask: does shadcn have this? If
close-but-not-quite, can I wrap it? Default to "yes, use shadcn."

## Architecture summary
- `App.tsx` owns a `useReducer<UiState>` state machine with four phases:
  `idle` (no job), `uploading` (XHR in flight), `rejoin` (server said
  409), `active` (job submitted, SSE open). Transitions are pure —
  `dispatch({type: ...})` only, no imperative state pokes.
- `<AppShell>` is a presentational two-column frame (400 px left, flex
  right, fixed 1080×760). It takes `left` and `right` slot children; it
  never touches domain state. Below 1080 px viewport width, it renders
  `<DesktopRequired>` instead.
- Non-active phases (`idle`, `uploading`, `rejoin`) render directly from
  `App` via `renderFileSlot / renderLanguagePairSlot / renderSubmitSlot`
  helpers that switch on `state.phase`. Each helper returns the correct
  left-column slot for that phase (Dropzone vs VideoCard, unlocked vs
  locked LanguagePair, idle vs uploading SubmitBar).
- The `active` phase delegates to a nested `<ActiveView>` that
  instantiates `useJobStream(jobId)` exactly once and threads the hook
  state into both the left SubmitBar and the right StageProgress +
  LogPanel + ResultPanel/FailureCard. Single SSE subscription per job —
  no double-subscribe via re-mount.
- `useJobStream` seeds from `GET /status`, subscribes to SSE, folds each
  event into reducer-style state, and emits an `activeStageElapsedMs`
  tick every second (driven by a `setInterval` bound to the current
  `stage_start.ts`; cleared on stage change, terminal event, or unmount).
- `createJob` is XHR-based so the `uploading` phase can surface real
  progress (`percent`, `bytesPerSec`, `etaSeconds`). `getHealth /
  getLanguages / getJobStatus / deleteJob` stay on `fetch`.
- 409 rejoin branch: when `createJob` rejects with `ApiError.status ===
  409`, App reads `err.concurrentJobDetail.active_job_id` and
  dispatches `uploadBlocked`. A `useEffect` on the rejoin phase fetches
  `getJobStatus(blockingJobId)` and dispatches `blockingStatusLoaded`.
  `<RejoinCard>` renders the fetched metadata + Rejoin CTA; clicking it
  transitions to `active` with `file === null` (no local preview for
  someone else's job).
- TS types in `src/api/schemas.ts` are a hand-mirrored copy of
  `server/app/schemas.py`. Do not drift (R7). OpenAPI codegen is the
  next move if the surface grows past ~10 models.

## Conventions
- TypeScript strict mode (`strict: true`, `noUnusedLocals`,
  `noUnusedParameters`, `noFallthroughCasesInSwitch`). SSE events use
  discriminated unions on `type`; narrow with `switch (ev.type)`.
- Path alias `@/*` → `src/*`. Configured in `tsconfig.json` +
  `vite.config.ts` + `vitest.config.ts` — keep all three in agreement.
- Components: `PascalCase.tsx`. Shared primitives in `src/components/`
  root (`Dropzone`, `StageProgress`, `LogPanel`, `ResultPanel`,
  `LanguageSelect`, `AppShell`, `DesktopRequired`). Phase-specific
  surfaces under `components/left/` (IdentityBlock, LeftColumn,
  LanguagePair, VideoCard, SubmitBar) and `components/right/`
  (StatusBand, IdlePlaceholder, UploadProgress, RejoinCard,
  FailureCard). shadcn primitives under `components/ui/`. Hooks:
  `useXxx.ts` in `src/hooks/`.
- Tests colocated under `__tests__/*.test.{ts,tsx}`. `src/setupTests.ts`
  loads `@testing-library/jest-dom/vitest` once per run; global env is
  `jsdom` (see `vitest.config.ts`).
- DOM-global names that clash are renamed in `api/schemas.ts`:
  `LogEvent` → `LogEventPayload`, `ErrorEvent` → `ErrorEventPayload`.
  The discriminator still lives on the `type` field, so consumers are
  unaffected.
- API paths live in `api/client.ts` only. A repo-wide grep for
  `/api/jobs/` should not return hits outside that file — use
  `outputUrl(jobId)` / `eventsUrl(jobId)` helpers everywhere else.

## Gotchas
- `tsc -b` runs as part of `npm run build` (project references). Any
  file listed in `tsconfig.node.json` (e.g. `vite.config.ts`,
  `vitest.config.ts`) must satisfy `composite: true` + no `noEmit`.
  If you add new non-src TS files consumed by build tooling, add them
  there, not to `tsconfig.json`.
- SSE via Vite proxy requires `cache-control: no-cache` on the upstream
  response — already wired in `vite.config.ts` via a `proxy.on('proxyRes')`
  handler. Intermediate caches otherwise buffer the stream and events
  arrive in bursts.
- `<video>` has no implicit ARIA role in jsdom; in tests use
  `container.querySelector("video")` rather than
  `getByRole("video"|"application")`.
- `<AppShell>` gates on `window.innerWidth >= 1080`. jsdom defaults to
  1024. Tests that render `<AppShell>` (or `<App>`) must set
  `Object.defineProperty(window, "innerWidth", { value: 1280 })` in
  `beforeAll` — otherwise `<DesktopRequired>` renders instead and every
  left/right-column assertion fails silently. See `App.test.tsx` for
  the pattern.
- `formatBytes` is inlined in two places (`App.tsx` and
  `components/right/UploadProgress.tsx`). Rule-of-three has not fired
  yet — if a third consumer appears, extract to `src/utils/`.

## Do NOT
- Don't duplicate language codes client-side. Always fetch from
  `GET /api/languages`. The server is the single source of truth (D12).
- Don't wrap renders in `try/catch` to catch React errors — use React
  Error Boundaries (not implemented in MVP, but noted). Fetch errors
  inside effects / handlers are fine to `try/catch`.
- Don't hardcode the API base URL. All requests use the `/api/*`
  relative prefix; Vite proxy (dev) and same-origin static mount (prod)
  both resolve it naturally. `api/client.ts` has `const BASE = "/api"`
  as the one place that word appears.
- Don't introduce a state-management library (Redux, Zustand, Jotai).
  MVP gets by with one `useReducer<UiState>` in `<App>` +
  `useJobStream` inside `<ActiveView>`. Reach for a library only when
  the local state machine outgrows hooks — not before.
- Don't copy shadcn CSS variables or duplicate the design tokens
  inline. Everything goes through `src/styles/globals.css`.

## References
- `/workspace/video-scene-text-translator/plan.md` — decisions D1–D19, risks R1–R10
- `/workspace/video-scene-text-translator/server/CLAUDE.md` — backend contract (SSE event shapes, API surface)
- Root `/workspace/video-scene-text-translator/CLAUDE.md` — project-wide conventions
