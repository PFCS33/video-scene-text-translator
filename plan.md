# Plan: AnyText2 Integration via Gradio API

## Goal
Integrate AnyText2 as a real Stage A text editing model, replacing the placeholder editor. AnyText2 runs as a separate Gradio server (managed by Hebin); our pipeline calls it via HTTP to perform style-preserving cross-language scene text replacement.

## Approach
Create an `AnyText2Editor` subclass of `BaseTextEditor` that communicates with AnyText2's Gradio server via `gradio_client`. The editor:
1. Takes a frontalized ROI image + target text from S3
2. Saves the ROI to a temp file (Gradio API needs file paths)
3. Generates a full-image mask (entire ROI is the edit region)
4. Calls `/process_1` (edit tab) with font set to "Mimic From Image" for style preservation
5. Downloads the first result image from the gallery response
6. Returns it as a BGR numpy array matching the original ROI dimensions

**Key decisions:**
- **Gradio client over raw HTTP**: `gradio_client` handles file upload, serialization, and result download. Less code, less bugs.
- **Full-image mask**: Since ROIs are already tight frontalized text crops, mask the entire image. Simpler than threshold-based text detection. Can refine later if background artifacts appear.
- **`img_count=1`**: Only generate one result to minimize latency (~1.2s instead of ~4.8s for 4 images).
- **Configurable server URL**: In `TextEditorConfig` so any teammate can point to their own server.
- **Connection validation**: `AnyText2Editor` checks server reachability on init and raises a clear error if the server is down.
- **Timeout handling**: Gradio calls can hang if the GPU is busy. Add a configurable timeout (default 60s).
- **Lazy init**: Follow existing pattern — don't connect until first `edit_text()` call.

**AnyText2 `/process_1` API mapping:**

| Our concept | API parameter | Value |
|---|---|---|
| Original image | `ori_img` | ROI saved as temp PNG |
| Image + mask | `ref_img` | `{background: ROI, layers: [white_mask]}` |
| Target text | `text_prompt` | e.g. `"咖啡"` |
| Scene description | `img_prompt` | `"Text with some background"` |
| Font style | `f1` | `"Mimic From Image(模仿图中字体)"` |
| Text color | `c1` | Auto-extracted from ROI border pixels |
| Image count | `img_count` | `1` |
| Dimensions | `w`, `h` | Match ROI dimensions (clamped to 256-1024) |
| Other fonts | `f2-f5`, `m2-m5`, `c2-c5` | Defaults (unused) |
| Model path | `base_model_path` | `""` (use server default) |
| LoRA | `lora_path_ratio` | `""` (none) |

## Files to Change
- [ ] `code/src/config.py` — Add `server_url`, `server_timeout`, `anytext2_ddim_steps`, `anytext2_cfg_scale`, `anytext2_strength` fields to `TextEditorConfig`
- [ ] (new) `code/src/models/anytext2_editor.py` — `AnyText2Editor(BaseTextEditor)`: Gradio client wrapper, mask generation, color extraction, temp file management
- [ ] `code/src/stages/s3_text_editing.py` — Register `"anytext2"` backend in `_init_editor()`, pass config to editor
- [ ] `code/config/default.yaml` — Add `server_url: null` and AnyText2 params to `text_editor` section
- [ ] `code/config/adv.yaml` — Same as default.yaml for text_editor section
- [ ] (new) `code/tests/test_anytext2_editor.py` — Unit tests with mocked Gradio client (no server needed)
- [ ] (new) `third_party/install_anytext2.sh` — Setup instructions: clone repo, create conda env, download weights, run server
- [ ] `code/requirements/base.txt` — Add `gradio_client` dependency

## Risks
- **Server availability**: AnyText2 server must be running for the editor to work. Pipeline will raise a clear error if it's down, and falls back to placeholder if configured.
- **Gradio API stability**: Gradio client versions can be finicky. Pin `gradio_client` version to match server's Gradio 5.12.0.
- **Image quality**: AnyText2's editing mode is under-evaluated in the paper. Quality on our specific ROIs (frontalized, cropped) is unknown until we test. Full-mask approach may cause background regeneration artifacts.
- **Latency**: ~1.2s per text track. Acceptable for reference-frame-only editing but would be a bottleneck if ever applied per-frame.
- **Network dependency**: Server is on `109.231.106.68` (lab network). Must be reachable from the machine running the pipeline.
- **ROI size constraints**: AnyText2 accepts 256-1024px. Very small or very large ROIs need resizing, which may affect quality.
- **Color extraction**: Auto-extracting text color from ROI border pixels is approximate. May not match AnyText2's expected format.

## Done When
- [ ] `AnyText2Editor.edit_text(roi, target_text)` returns a style-preserved edited ROI when server is running
- [ ] Pipeline runs end-to-end with `text_editor.backend: "anytext2"` and produces output video with translated text
- [ ] Editor raises clear error message when server is unreachable
- [ ] Editor handles edge cases: empty ROI, very small ROI (<256px), very large ROI (>1024px)
- [ ] All existing tests pass (zero regressions)
- [ ] New tests cover: mock Gradio call, mask generation, color extraction, error handling, ROI resizing
- [ ] Code review approved (@reviewer)
- [ ] Changes committed as atomic commits

## Progress
- [x] Step 1: Add AnyText2 config fields to `TextEditorConfig` in `config.py`
- [x] Step 2: Implement `AnyText2Editor` in `code/src/models/anytext2_editor.py`
- [x] Step 3: Register `"anytext2"` backend in `s3_text_editing.py`
- [x] Step 4: Update `default.yaml` and `adv.yaml` with new text_editor fields
- [x] Step 5: Add `gradio_client` to `requirements/base.txt`
- [x] Step 6: Write unit tests with mocked Gradio client (15 tests, all passing)
- [x] Step 7: Write `third_party/install_anytext2.sh` setup script
- [ ] Step 8: Integration test with live server (manual, with Hebin)
- [x] Step 9: Update CHANGELOG.md with AnyText2 integration entry
