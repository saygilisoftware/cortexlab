# Vendored Three.js provenance

Version: Three.js **0.170.0**, official tag **r170**. License: MIT, copyright 2010-2024 Three.js authors; full text in [THREE-LICENSE.txt](THREE-LICENSE.txt). Retrieved 2026-11-06. This version is intentionally pinned; it is not represented as the latest release.

| Local file | Tagged official source | SHA-256 | UTF-8 bytes / lines |
| --- | --- | --- | --- |
| `three.module.js` | [build/three.module.js](https://raw.githubusercontent.com/mrdoob/three.js/r170/build/three.module.js) | `27544467e1c185013d9f2c98ebcb3678980c97ef737eeaee59c3a87552c69bf5` | 1,312,069 / 51,959 |
| `OrbitControls.js` | [examples/jsm/controls/OrbitControls.js](https://raw.githubusercontent.com/mrdoob/three.js/r170/examples/jsm/controls/OrbitControls.js) | `17541c43f70d5b522c3d511add10b480e15420b1fac57f52464f2e7b15f49fdd` | 32,065 / 1,454 |
| `THREE-LICENSE.txt` | [LICENSE](https://raw.githubusercontent.com/mrdoob/three.js/r170/LICENSE) | `da0f7685eaab80e330ad167aa59b2620419adc568e92a7ef2e1bc447f41ffcbc` | 1,079 / 19 |

## Retrieval and modifications

Shell and browser network access were unavailable. The connected search-service tool retrieved official raw GitHub source as numbered text windows. Windows were merged by source-line number; missing intervals were requested until all 51,966 search-presentation lines were covered. Source was written with UTF-8 and LF endings, retaining the license header.

The search presentation artificially split three compiled shader strings across seven extra lines at decimal expressions. Those strings were recovered from complete official same-tag sources: [lights_fragment_begin.glsl.js](https://raw.githubusercontent.com/mrdoob/three.js/r170/src/renderers/shaders/ShaderChunk/lights_fragment_begin.glsl.js), [packing.glsl.js](https://raw.githubusercontent.com/mrdoob/three.js/r170/src/renderers/shaders/ShaderChunk/packing.glsl.js), and [tonemapping_pars_fragment.glsl.js](https://raw.githubusercontent.com/mrdoob/three.js/r170/src/renderers/shaders/ShaderChunk/tonemapping_pars_fragment.glsl.js). The transformation follows the official [rollup GLSL transform](https://raw.githubusercontent.com/mrdoob/three.js/r170/utils/build/rollup.config.js): trim the template body, remove carriage returns and comments, collapse repeated newlines, and JSON-stringify. No shader implementation was invented. The resulting build has 51,959 lines. The hashes above identify the delivered files; they are not asserted to be independently verified upstream release hashes.

The official r170 `OrbitControls.js` already imports from the bare specifier `three`; this import is retained and resolves through the browser import map. No import-path edit was needed. No application code from unrelated projects was used.

## Checks performed

- `node --check frontend/vendor/three.module.js` passed.
- `node --check frontend/vendor/OrbitControls.js` passed.
- Dynamic module import reported revision `170`, 415 exports, and `Vector3(1,2,3).length()` = `3.7416573867739413`.

These checks establish parsing and basic module integrity. Browser rendering/shader checks are recorded separately in the project's validation report.

## Independent Git blob verification attempt

On 2026-11-06 the official GitHub contents endpoints for both files were requested with `?ref=r170`, together with the repository git-trees endpoint for `r170` (recursive and nonrecursive) and the build-directory contents endpoint. Both available web providers reported these API URLs inaccessible. Tagged GitHub HTML pages were accessible but exposed no file blob SHA in the returned representation. Therefore byte-for-byte identity with official Git blobs could not be independently established; no upstream checksum match is claimed.

Local Git blob IDs were computed as `SHA1(b"blob " + decimal_byte_length + b"\0" + file_bytes)`:

| File | Local Git blob SHA-1 |
| --- | --- |
| `three.module.js` | `333b94fd889258cf37ff7c2f227aeaac8348016d` |
| `OrbitControls.js` | `cc8bd772ad4ff254ff85eedcfd74a4cddf226f1e` |

These complement the SHA-256 hashes above and permit a future exact comparison when the official API or a tagged checkout is available. The documented shader reconstruction and line-ending normalization remain relevant potential sources of byte differences even though parsing and real application rendering have been validated separately.
