# Releasing ttucs to PyPI

ttucs ships as **two** PyPI distributions:

| distribution | import name | contents | version |
|---|---|---|---|
| `ttucs` | `ttucs` | code, UCS list, CLI | follows the code (`src/ttucs/_version.py`) |
| `ttucs-model-bow` | `ttucs_model_bow` | data only: `config.json`, fp16 `weights.safetensors`, `manifest.json` (SHA-256) | follows the weights (`packages/ttucs-model-bow/pyproject.toml`) |

`ttucs` depends on `ttucs-model-bow>=0.1.1,<0.2`, so `pip install ttucs` installs the weights too. Wheels
cannot run code at install time, which is why the weights live in their own distribution. After that,
inference is offline.

The two versions are independent. Bump `ttucs-model-bow` only when the weights change, then update
`ttucs.hub.MODELS["bow"]` (version + sha256), `packages/ttucs-model-bow/RELEASE_SHA256` and the
dependency range in `pyproject.toml`. The weights are **never committed to git**. They travel as a
GitHub Release asset (`model-bow-v<version>.tar.gz`) that CI downloads and verifies by SHA-256.

> Never paste API tokens into issues, commits, chat or shell history. Trusted Publishing (below) needs no
> token at all.

## One-time setup

1. **Accounts.** Create accounts on <https://pypi.org> and <https://test.pypi.org> (they are separate
   sites with separate accounts). Enable 2FA on both: PyPI requires it for uploads.
2. **Enable the workflows.** The assistant's GitHub App cannot write `.github/workflows/`, so move the
   files yourself:
   ```bash
   mkdir -p .github/workflows          # PowerShell: New-Item -ItemType Directory -Force .github/workflows
   git mv ci/github-actions-ci.yml .github/workflows/ci.yml
   git mv ci/publish.yml .github/workflows/publish.yml
   git commit -m "ci: enable CI and publish workflows"
   git push
   ```
   Commands in this guide are bash unless a PowerShell version is given. On Windows PowerShell 5.1,
   run commands joined by `&&` one line at a time, and join backslash-continued lines into one line.
3. **GitHub environments.** In the repo, go to *Settings → Environments* and create `pypi` and
   `testpypi`. For `pypi`, adding yourself as a required reviewer is recommended.
4. **Pending trusted publishers (4 forms in total).** This repo publishes two PyPI projects, `ttucs`
   and `ttucs-model-bow`. PyPI ties a trusted publisher to one project name, so each project needs its
   own registration, on both PyPI and TestPyPI:

   | # | Site | Where | PyPI project name | Environment |
   |---|---|---|---|---|
   | 1 | pypi.org | *Your account → Publishing → Add a new pending publisher* | `ttucs` | `pypi` |
   | 2 | pypi.org | same page, submit the form again | `ttucs-model-bow` | `pypi` |
   | 3 | test.pypi.org | same page on TestPyPI (separate account) | `ttucs` | `testpypi` |
   | 4 | test.pypi.org | same page, submit the form again | `ttucs-model-bow` | `testpypi` |

   The other fields are identical in all four forms:

   | field | value |
   |---|---|
   | Owner | `SoundsLikeJonny` |
   | Repository name | `ttucs` |
   | Workflow name | `publish.yml` |

   A pending publisher turns into a normal one, and the project is created, on the first upload. All four
   point to the same workflow, so one run of `publish.yml` can upload both packages. Without the
   `ttucs-model-bow` registration, that upload is rejected with an `invalid-publisher` error.

## Release steps (example: ttucs 0.2.0 with ttucs-model-bow 0.1.1)

1. Merge the packaging PR (#2) into `main`.
2. **Build the weights** (on any machine with the dev install):
   ```bash
   pip install -e packages/ttucs-model-bow -e ".[dev]"
   ttucs generate-data -c configs/bow-release.yaml
   ttucs train -c configs/bow-release.yaml                       # compact, redistributable data only
   python scripts/build_model_package.py --model-dir models/ttucs-bow-release --tarball --wheel
   ```
   This writes `dist/model-bow-v0.1.1.tar.gz` and its `.sha256`. If you retrain, the sha256 changes:
   put the new value in `packages/ttucs-model-bow/RELEASE_SHA256` **and** `src/ttucs/hub.py`
   (`MODELS["bow"].sha256`) and commit. `tests/test_hub.py` checks that the two match.
3. **Upload the model asset.** Create a GitHub Release with tag `model-bow-v0.1.1`. It must not start
   with `v`, so it does not trigger the publish workflow. Attach `model-bow-v0.1.1.tar.gz`. The same
   URL serves as the download fallback for `ttucs download`.
   ```bash
   gh release create model-bow-v0.1.1 dist/model-bow-v0.1.1.tar.gz --title "ttucs-model-bow 0.1.1 weights" --notes "Release weights (fp16 safetensors)"
   ```
4. **TestPyPI.** Go to *Actions → Publish → Run workflow* on `main`. Then check the install in a clean
   environment:
   macOS / Linux (bash, zsh):
   ```bash
   python -m venv /tmp/tt && . /tmp/tt/bin/activate
   pip install torch --index-url https://download.pytorch.org/whl/cpu
   pip install -i https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ ttucs==0.2.0
   TTUCS_OFFLINE=1 ttucs info && TTUCS_OFFLINE=1 ttucs predict "wooden door creak slow, take 3"
   ```
   Windows (PowerShell). Windows PowerShell 5.1 has no `&&`, the venv script lives in `Scripts\`, and
   environment variables are set with `$env:`:
   ```powershell
   python -m venv "$env:TEMP\tt"
   & "$env:TEMP\tt\Scripts\Activate.ps1"
   pip install torch --index-url https://download.pytorch.org/whl/cpu
   pip install -i https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ ttucs==0.2.0
   $env:TTUCS_OFFLINE = "1"
   ttucs info
   ttucs predict "wooden door creak slow, take 3"
   Remove-Item Env:TTUCS_OFFLINE
   ```
   If activation fails with "running scripts is disabled on this system", run
   `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` first. It only applies to that window.
5. **PyPI.** Tag the merged commit and push the tag. The workflow checks that the tag matches
   `ttucs.__version__`.
   ```bash
   git checkout main
   git pull
   git tag -a v0.2.0 -m "ttucs 0.2.0"
   git push origin v0.2.0
   ```
   `skip-existing` lets a later ttucs release reuse an unchanged `ttucs-model-bow` without failing.
6. Create the GitHub Release for `v0.2.0`, using the CHANGELOG section as notes.

## Manual alternative (API token)

Use this only if you don't want Trusted Publishing:

1. Create a **project-scoped** API token on PyPI/TestPyPI. For the first upload of a new project it has
   to be account-scoped; replace it with a project-scoped one afterwards.
2. Store it where only you can read it. Either `~/.pypirc` (`chmod 600 ~/.pypirc`):
   ```ini
   [pypi]
   username = __token__
   password = <your pypi token>

   [testpypi]
   repository = https://test.pypi.org/legacy/
   username = __token__
   password = <your testpypi token>
   ```
   or environment variables set in your shell profile or secret manager (`TWINE_USERNAME=__token__`,
   `TWINE_PASSWORD=<token>`). Never `echo` them.
3. Build and upload:
   ```bash
   rm -rf dist && python scripts/build_model_package.py --model-dir models/ttucs-bow-release --wheel
   python -m build --outdir dist .
   twine check --strict dist/*.whl dist/ttucs-*.tar.gz dist/ttucs_model_bow-*.tar.gz
   twine upload -r testpypi dist/*.whl dist/ttucs-*.tar.gz dist/ttucs_model_bow-*.tar.gz   # then: twine upload ...
   ```
   Do not upload `model-bow-v*.tar.gz`: that is the GitHub Release asset, not a Python distribution.
