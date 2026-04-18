/**
 * <IdlePlaceholder> — empty-state surface shown in the right column when
 * no job is in flight. Purely static; no props, no state.
 *
 * Mockup reference: `.idle-wrap` + `.idle-icon` in mockup.html. A 72×72
 * circular dashed border holds an inline SVG of nested squares (outer
 * rounded rect + inner square). Matches the mockup pixel-for-pixel; the
 * previous lucide UploadCloud was a best-effort placeholder, now dropped.
 */

export function IdlePlaceholder(): JSX.Element {
  return (
    <div className="flex flex-1 items-center justify-center p-8">
      <div className="flex flex-col items-center gap-4">
        <div
          aria-hidden
          className="flex h-[72px] w-[72px] items-center justify-center rounded-full border-[1.5px] border-dashed border-[color:var(--line-2)] text-[color:var(--ink-3)]"
        >
          <svg
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            strokeWidth={1.6}
            strokeLinecap="round"
            strokeLinejoin="round"
            className="h-8 w-8"
          >
            <rect x={3} y={3} width={18} height={18} rx={2} />
            <path d="M9 9h6v6H9z" />
          </svg>
        </div>
        <p className="font-mono text-[11px] font-semibold uppercase tracking-[0.16em] text-[color:var(--ink-2)]">
          Waiting for a job
        </p>
        <p className="max-w-[42ch] text-center text-sm leading-relaxed text-[color:var(--ink-1)]">
          Pick a file and hit <b>Start translation</b> on the left. Stages,
          log, and the result will all appear in this window.
        </p>
      </div>
    </div>
  );
}
