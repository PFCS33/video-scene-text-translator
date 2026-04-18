/**
 * <UploadForm> — composite upload form with drop-target, two language
 * selects, and a submit button. Owns its own file/lang/error state for
 * this MVP step; the wider job state machine lands in Step 13 via
 * `useJobStream`.
 *
 * Submit flow:
 *   - Calls `createJob(file, source, target)`.
 *   - On success: `onJobCreated(job_id)`.
 *   - On ApiError: switches into an error state whose kind steers the
 *     alert copy (409 → concurrent job + rejoin, 413 → size, 400 → lang).
 *
 * Disabled-guards on submit:
 *   1. no file picked
 *   2. either language empty (shouldn't happen after languages load, but
 *      guard anyway for the brief mount window)
 *   3. source === target  (with a helper-text warning — per spec, either
 *      disable or warn is fine; we do both so it's obvious why)
 *   4. a request is in flight
 */

import { useCallback, useEffect, useState } from "react";
import { Loader2 } from "lucide-react";

import {
  Alert,
  AlertDescription,
  AlertTitle,
} from "@/components/ui/alert";
import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardFooter,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";

import { ApiError, createJob, getLanguages } from "@/api/client";
import type { Language } from "@/api/schemas";

import { Dropzone } from "./Dropzone";
import { LanguageSelect } from "./LanguageSelect";

const DEFAULT_SOURCE = "en";
const DEFAULT_TARGET = "es";

export interface UploadFormProps {
  onJobCreated: (jobId: string) => void;
  /** Wired up by Step 13; invoked when the user clicks "Rejoin" on a 409. */
  onRejoinActiveJob?: (jobId: string) => void;
}

type SubmitError =
  | { kind: "concurrent"; message: string; activeJobId: string | null }
  | { kind: "size"; message: string }
  | { kind: "lang"; message: string }
  | { kind: "other"; message: string };

function errorTitle(error: SubmitError): string {
  switch (error.kind) {
    case "concurrent":
      return "A job is already running";
    case "size":
      return "File too large";
    case "lang":
      return "Invalid language code";
    default:
      return "Upload failed";
  }
}

function renderError(
  error: SubmitError,
  onRejoinActiveJob?: (jobId: string) => void,
) {
  // Pull activeJobId into a local const so TypeScript can narrow it to
  // `string` through the closure — avoids an `as string` cast in the JSX.
  const activeJobId =
    error.kind === "concurrent" ? error.activeJobId : null;

  return (
    <Alert variant="destructive">
      <AlertTitle>{errorTitle(error)}</AlertTitle>
      <AlertDescription className="space-y-2">
        <p>{error.message}</p>
        {error.kind === "concurrent" && (
          <p>
            {activeJobId
              ? `Active job id: ${activeJobId}. Rejoin it or cancel it first.`
              : "Cancel it first or wait for it to finish."}
          </p>
        )}
        {activeJobId && (
          <Button
            type="button"
            variant="outline"
            size="sm"
            onClick={() => onRejoinActiveJob?.(activeJobId)}
          >
            Rejoin
          </Button>
        )}
      </AlertDescription>
    </Alert>
  );
}

function errorFromApi(err: unknown): SubmitError {
  if (err instanceof ApiError) {
    if (err.status === 409) {
      const detail = err.concurrentJobDetail;
      return {
        kind: "concurrent",
        activeJobId: detail?.active_job_id ?? null,
        message: "A job is already running on this server.",
      };
    }
    if (err.status === 413) {
      return {
        kind: "size",
        message: "File too large (the server caps uploads at 200 MB).",
      };
    }
    if (err.status === 400) {
      return {
        kind: "lang",
        message:
          typeof err.detail === "string"
            ? `Invalid language code: ${err.detail}`
            : "Invalid language code.",
      };
    }
    return {
      kind: "other",
      message:
        typeof err.detail === "string"
          ? `Upload failed: ${err.detail}`
          : `Upload failed (status ${err.status}).`,
    };
  }
  return {
    kind: "other",
    message: `Upload failed: ${(err as Error)?.message ?? "unknown error"}`,
  };
}

export function UploadForm({
  onJobCreated,
  onRejoinActiveJob,
}: UploadFormProps) {
  const [file, setFile] = useState<File | null>(null);
  const [languages, setLanguages] = useState<Language[]>([]);
  const [sourceLang, setSourceLang] = useState<string>(DEFAULT_SOURCE);
  const [targetLang, setTargetLang] = useState<string>(DEFAULT_TARGET);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [error, setError] = useState<SubmitError | null>(null);

  useEffect(() => {
    let cancelled = false;
    getLanguages()
      .then((langs) => {
        if (cancelled) return;
        setLanguages(langs);
        // If the curated defaults aren't in the list (shouldn't happen in
        // prod, but be robust), fall back to the first two codes.
        const codes = new Set(langs.map((l) => l.code));
        if (!codes.has(DEFAULT_SOURCE) && langs[0]) {
          setSourceLang(langs[0].code);
        }
        if (!codes.has(DEFAULT_TARGET) && langs[1]) {
          setTargetLang(langs[1].code);
        }
      })
      .catch((e) => {
        if (cancelled) return;
        setError({
          kind: "other",
          message: `Could not load language list: ${
            (e as Error)?.message ?? "unknown error"
          }`,
        });
      });
    return () => {
      cancelled = true;
    };
  }, []);

  const sameLang = sourceLang === targetLang && sourceLang !== "";
  const canSubmit =
    !isSubmitting &&
    file !== null &&
    sourceLang !== "" &&
    targetLang !== "" &&
    !sameLang;

  const handleSubmit = useCallback(async () => {
    if (!canSubmit || !file) return;
    setError(null);
    setIsSubmitting(true);
    try {
      const { job_id } = await createJob(file, sourceLang, targetLang);
      onJobCreated(job_id);
    } catch (e) {
      setError(errorFromApi(e));
    } finally {
      setIsSubmitting(false);
    }
  }, [canSubmit, file, sourceLang, targetLang, onJobCreated]);

  return (
    <Card className="w-full max-w-xl">
      <CardHeader>
        <CardTitle>Scene Text Translator</CardTitle>
        <CardDescription>
          Upload a video, pick source and target languages, and the
          five-stage pipeline will replace on-screen text across languages.
        </CardDescription>
      </CardHeader>

      <CardContent className="space-y-4">
        <Dropzone
          currentFile={file}
          onFileSelected={setFile}
          disabled={isSubmitting}
        />

        <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
          <LanguageSelect
            label="Source language"
            value={sourceLang}
            onChange={setSourceLang}
            languages={languages}
            disabled={isSubmitting || languages.length === 0}
          />
          <LanguageSelect
            label="Target language"
            value={targetLang}
            onChange={setTargetLang}
            languages={languages}
            disabled={isSubmitting || languages.length === 0}
          />
        </div>

        {sameLang && (
          <p className="text-sm text-destructive">
            Source and target must differ.
          </p>
        )}

        {error && renderError(error, onRejoinActiveJob)}
      </CardContent>

      <CardFooter>
        <Button
          type="button"
          className="w-full"
          disabled={!canSubmit}
          onClick={handleSubmit}
        >
          {isSubmitting && (
            <Loader2 className="mr-2 h-4 w-4 animate-spin" aria-hidden />
          )}
          {isSubmitting ? "Uploading…" : "Start translation"}
        </Button>
      </CardFooter>
    </Card>
  );
}
