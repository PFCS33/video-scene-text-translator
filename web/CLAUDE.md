# web/ — React frontend for the live demo

Nested CLAUDE.md — auto-loads when working on the React SPA. Root CLAUDE.md
covers the pipeline; this file only covers the frontend.

## Scope
React 18 + Vite + TypeScript SPA that drives the FastAPI server in
`server/`. Single view toggles between `<UploadForm>` and `<JobView>`.
Upload video → pick languages → watch per-stage progress + live log via
SSE → download the output MP4.

## Run
```bash
source /opt/nvm/nvm.sh              # Node >= 20 via nvm
cd web
npm install                         # or npm ci for reproducible installs
npm run dev                         # Vite :5173, proxies /api to :8000
npm run test                        # vitest + jsdom (51 tests)
npm run build                       # tsc -b && vite build → dist/
npm run type-check                  # tsc --noEmit
```

## Component-reuse principle (D6) — in priority order
1. **shadcn primitives first.** Use `<Button>`, `<Select>`, `<Label>`,
   `<Card>`, `<Alert>`, `<Input>`, `<Progress>`, `<Badge>` from
   `components/ui/*` as-is. Add more with
   `npx shadcn@latest add <name>` (config lives in `components.json`).
2. **Wrap-and-extend second.** If a primitive needs a thin domain shim
   (e.g. `LanguageSelect` wraps `<Select>` + `<Label>` and pre-fills
   options from `/api/languages`), keep the wrapper stateless:
   controlled props in, shadcn primitive out. No hidden state, no style
   overrides.
3. **Custom from scratch last.** Only when shadcn has no equivalent —
   `Dropzone`, `StageProgress` (5 discrete pills doesn't fit shadcn's
   single continuous `<Progress>` bar), `LogPanel`, `ErrorAlert`. Even
   then, use Tailwind classes bound to the design tokens in
   `src/styles/globals.css` (`bg-background`, `text-muted-foreground`,
   `border-border`, `bg-primary`) so custom components stay coherent.

Before adding a custom component, ask: does shadcn have this? If
close-but-not-quite, can I wrap it? Default to "yes, use shadcn."

## Architecture summary
- `App.tsx` holds a single `useState<string | null>(activeJobId)` and
  swaps between `<UploadForm>` (idle) and `<JobView jobId={...}>`
  (submitted). No router, no state-management library.
- `UploadForm` fetches `/api/languages` on mount, builds the form, calls
  `createJob(file, src, tgt)`. A 409 response unwraps into a
  `ConcurrentJobErrorDetail` via `ApiError.concurrentJobDetail`, rendered
  as an Alert with a Rejoin button (R8).
- `JobView` delegates lifecycle to the `useJobStream(jobId)` hook —
  seeds initial state from `getJobStatus`, subscribes to SSE via
  `openEventStream`, folds each `SSEEvent` into state through a
  `setState(prev => switch(ev.type))` reducer.
- On SSE disconnect the `openEventStream` helper auto re-syncs via
  `getJobStatus` so the stage progress stays correct across the gap
  (D16). Log lines during the gap are accepted as lost.
- TS types in `src/api/schemas.ts` are a hand-mirrored copy of
  `server/app/schemas.py`. Do not drift (R7). OpenAPI codegen is the
  next move if the surface grows past ~10 models.

## Conventions
- TypeScript strict mode (`strict: true`, `noUnusedLocals`,
  `noUnusedParameters`, `noFallthroughCasesInSwitch`). SSE events use
  discriminated unions on `type`; narrow with `switch (ev.type)`.
- Path alias `@/*` → `src/*`. Configured in `tsconfig.json` +
  `vite.config.ts` + `vitest.config.ts` — keep all three in agreement.
- Components: `PascalCase.tsx` in `src/components/` (shadcn primitives
  under `components/ui/`). Hooks: `useXxx.ts` in `src/hooks/`.
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
- Radix primitives (underpinning shadcn `<Select>`) call
  `hasPointerCapture`, `releasePointerCapture`, and `scrollIntoView` on
  DOM elements. jsdom lacks these — stub them in `beforeAll` in any
  test that opens a Radix popper (see
  `LanguageSelect.test.tsx` / `UploadForm.test.tsx` for the pattern).
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
  MVP gets by with one `useState<string | null>` in `<App>` +
  `useJobStream` inside `<JobView>`. Reach for a library only when the
  local state machine outgrows hooks — not before.
- Don't copy shadcn CSS variables or duplicate the design tokens
  inline. Everything goes through `src/styles/globals.css`.

## References
- `/workspace/video-scene-text-translator/plan.md` — decisions D1–D19, risks R1–R10
- `/workspace/video-scene-text-translator/server/CLAUDE.md` — backend contract (SSE event shapes, API surface)
- Root `/workspace/video-scene-text-translator/CLAUDE.md` — project-wide conventions
