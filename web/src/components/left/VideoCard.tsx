/**
 * <VideoCard> — the picked-file preview surface used in the left column
 * once a user has selected a video. Shows a native <video controls> playing
 * an object-URL blob, with a corner tag (INPUT / INPUT · EN / YOUR INPUT ·
 * QUEUED) and a filename + size footer in mono type.
 *
 * Design decision D3: "minimal video preview" — no custom scrubber, no
 * fps/resolution probe. The browser's own controls bar is the affordance.
 *
 * Risk mitigation R4: `URL.createObjectURL` leaks the underlying blob until
 * `URL.revokeObjectURL` runs. A `useEffect` cleanup handles that when the
 * component unmounts or the file reference changes.
 *
 * The `↻ replace` chip from the mockup is deferred per plan.md (defer list).
 * We don't render a replace button; the caller swaps this card out at the
 * phase boundary instead.
 */

import { useEffect, useState } from "react";

export type VideoCardVariant = "input" | "queued";

export interface VideoCardProps {
  file: File;
  /** "input" → INPUT tag; "queued" → YOUR INPUT · QUEUED */
  variant: VideoCardVariant;
  /** When set on variant="input", renders "INPUT · <SOURCELANG>". Ignored on "queued". */
  sourceLang?: string;
}

function formatMB(bytes: number): string {
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

function cornerTagText(
  variant: VideoCardVariant,
  sourceLang?: string,
): string {
  if (variant === "queued") return "YOUR INPUT · QUEUED";
  if (sourceLang && sourceLang.length > 0) {
    return `INPUT · ${sourceLang.toUpperCase()}`;
  }
  return "INPUT";
}

export function VideoCard({
  file,
  variant,
  sourceLang,
}: VideoCardProps): JSX.Element {
  // Blob URL lifecycle. We hold the URL in state (not useMemo) so the
  // initial render can guard on `blobUrl && ...` and avoid a paint with
  // src="" — some browsers refuse to load after that.
  //
  // Revoke timing: `queueMicrotask` defers the revoke past React's
  // synchronous cleanup pass. Why this matters under React 18 StrictMode
  // (dev): React runs `mount → cleanup → mount` synchronously to surface
  // impure effects. If we revoked synchronously in cleanup, the `<video>`
  // tag would still have `src=<URL1>` in the DOM (the second mount hasn't
  // swapped state to URL2 yet), and we'd yank the blob out from under it.
  // Deferring the revoke a microtask gives React room to commit the new
  // state before the tear-down runs.
  //
  // On genuine unmount (and in production where StrictMode doesn't
  // double-invoke) the microtask still fires — just a tick later than it
  // would have. No leak (R4).
  const [blobUrl, setBlobUrl] = useState<string | null>(null);
  useEffect(() => {
    const url = URL.createObjectURL(file);
    setBlobUrl(url);
    return () => {
      queueMicrotask(() => URL.revokeObjectURL(url));
    };
  }, [file]);

  const tag = cornerTagText(variant, sourceLang);

  return (
    <div className="flex flex-col gap-2">
      <div
        className="relative overflow-hidden rounded-md border border-[color:var(--line-2)] bg-black aspect-video"
      >
        {/* aria-hidden: the corner tag is cosmetic. The video's aria-label
            carries the filename for assistive tech. */}
        <div
          aria-hidden
          className="pointer-events-none absolute top-2 left-2 z-10 rounded-sm bg-black/70 px-[7px] py-1 font-mono text-[9px] uppercase tracking-wider text-[color:var(--ink-1)] backdrop-blur"
        >
          {tag}
        </div>
        {blobUrl && (
          <video
            controls
            src={blobUrl}
            aria-label={file.name}
            className="block h-full w-full object-contain"
          />
        )}
      </div>
      <div className="flex items-center justify-between gap-2 font-mono text-[10px] tracking-wide text-[color:var(--ink-3)]">
        <span className="truncate">{file.name}</span>
        <span className="shrink-0">{formatMB(file.size)}</span>
      </div>
    </div>
  );
}
