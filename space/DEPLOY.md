# Deploying the demo to Google Cloud Run

The demo is a container that runs the same `production_rag` package the evaluation
measured and the same `space/app.py`. Nothing about retrieval or generation changes
because it is deployed; this document is purely operational.

> **On cost.** Cloud Run has an always-free monthly allowance that Google describes as
> non-expiring, but it is **subject to change and requires billing to be enabled on the
> project**. This is not a sandbox: usage beyond the allowance is charged to a real card.
> Set a budget alert before deploying, keep `--min-instances 0`, and treat the numbers
> below as configuration rather than a guarantee of a zero bill.

## Prerequisites

1. **`gcloud` CLI** — <https://cloud.google.com/sdk/docs/install>, then `gcloud init`.
2. **A GCP project with billing enabled.**
3. **APIs enabled** on that project:
   ```bash
   gcloud services enable run.googleapis.com artifactregistry.googleapis.com secretmanager.googleapis.com
   ```

## 1. Build

Build from the **repository root** — the image needs `pyproject.toml` and `src/`:

```bash
docker build -f space/Dockerfile -t production-rag:local .
```

The image bakes in the pinned Hugging Face dataset index and the `bge-small-en-v1.5`
encoder, so a cold start does not download 61 MB before it can serve. See the Dockerfile
header for why that matters against Cloud Run's 240 s startup probe.

Run it locally exactly as Cloud Run will:

```bash
docker run --rm -p 8080:8080 -e PORT=8080 production-rag:local
# then open http://localhost:8080
```

## 2. The secret — you create this, not the deploy script

`OPENROUTER_API_KEY` is **not** in the image and **not** in this repository. Without it the
service serves retrieval and states plainly that generation is disabled, which is a valid
way to run it.

To enable generation, create the secret yourself and grant the runtime service account
access:

```bash
# Paste the key at the prompt; it never lands in shell history or a file.
gcloud secrets create openrouter-api-key --replication-policy=automatic --data-file=-
```

Then reference it at deploy time with `--set-secrets` (below). Do not pass the key with
`--set-env-vars`: that stores it in the service's plaintext configuration.

## 3. Deploy

```bash
PROJECT=$(gcloud config get-value project)
REGION=europe-west1          # pick one near you

gcloud run deploy production-rag \
  --source . \
  --region "$REGION" \
  --allow-unauthenticated \
  --memory 2Gi \
  --cpu 2 \
  --concurrency 80 \
  --min-instances 0 \
  --max-instances 2 \
  --timeout 300 \
  --set-secrets OPENROUTER_API_KEY=openrouter-api-key:latest
```

`--source .` uses Cloud Build and needs the Dockerfile at a path it detects; if it does not
pick up `space/Dockerfile`, build and push explicitly instead:

```bash
gcloud auth configure-docker "${REGION}-docker.pkg.dev"
docker tag production-rag:local "${REGION}-docker.pkg.dev/${PROJECT}/demos/production-rag:v1"
docker push "${REGION}-docker.pkg.dev/${PROJECT}/demos/production-rag:v1"
gcloud run deploy production-rag \
  --image "${REGION}-docker.pkg.dev/${PROJECT}/demos/production-rag:v1" \
  --region "$REGION" --allow-unauthenticated \
  --memory 2Gi --cpu 2 --concurrency 80 --min-instances 0 --max-instances 2 --timeout 300 \
  --set-secrets OPENROUTER_API_KEY=openrouter-api-key:latest
```

## Why this configuration

| Flag | Value | Reason |
|---|---|---|
| `--memory` | `2Gi` | torch runtime, the encoder, 22,789 chunk texts and a 35 MB vector matrix all live in the process. 1 Gi is too tight; 2 Gi leaves headroom for a request. |
| `--cpu` | `2` | Encoder inference and model load are CPU-bound. 2 vCPU roughly halves cold start against 1. |
| `--concurrency` | `80` (default) | **Corrected after the first deployment.** The original value here was `4`, reasoned from "these are CPU-bound encodes". That conflated HTTP concurrency with compute concurrency: a Gradio page load fires ~65 parallel static-asset requests, so with one instance up during a cold start, half the bundle returned **429** and the page hung on "Loading...". The edge must stay wide; compute is limited in-process by `demo.queue(default_concurrency_limit=2)`. |
| `--min-instances` | `0` | Scale to zero. This is what keeps idle cost at zero, and the price is a cold start on the first request after idle. |
| `--max-instances` | `2` | A cap is a spend limit. A public URL with unbounded autoscaling is an unbounded bill. |
| `--timeout` | `300` | A generation call can take well over a minute on a free model. |
| `--allow-unauthenticated` | — | It is a public demo. |
| `--set-secrets` | — | Keeps the key out of the image, the repo and the service's plaintext env. |

## Verification checklist

After deploying, check each of these against the public URL and record the result:

- [ ] URL returns the Gradio UI
- [ ] Retrieval: "how do I read a parquet file in duckdb?" ranks a DuckDB Parquet page first
- [ ] Citations: an answer's `[n]` markers link to the passages listed under Sources
- [ ] Refusal: the "quantum flux capacitor" example is refused, not answered
- [ ] Generation-disabled mode: with no secret bound, the banner says so and retrieval works
- [ ] Generation-enabled mode: with the secret bound, an answer is produced
- [ ] Cold start and warm response times **measured**, not estimated

Every item above **except the Cloud Run-specific ones** was already verified against the
container locally — see the measured table below and the session notes in `PROGRESS.md`.

## Measured startup

All measured on this machine (Docker 29.7.2, Linux containers, `--memory 2g --cpus 2`,
image already present locally), not estimated.

| Phase | Measured |
|---|---|
| Image size (uncompressed) | **2.78 GB** |
| — of which `pip install` layer | 1.61 GB (CPU torch, transformers, gradio 6, scipy) |
| — of which baked index | 61 MB |
| — of which baked encoder | 129 MB |
| Container start → first HTTP 200 | **62 s** |
| Warm page response | **0.008 s** |
| Warm query, retrieval only | < 1 s |
| Warm query, generation (`ollama-qwen`, 7B on CPU) | 19–39 s |
| **Cloud Run cold start** (rev `00002-dtq`, `asia-south1`) | **39.2 s** server-side on `/`, 39.9 s browser TTFB |
| **Cloud Run warm response** | **0.065 s** |

The Cloud Run cold figure is a real scale-to-zero cold start — the instance-start log line reads
`Reason: AUTOSCALING`, and the request that triggered it arrived ~20 ms earlier. It includes the
2.78 GB image pull, which the local container's 62 s figure excludes — and it is nonetheless the
*smaller* number, because Cloud Run applies startup CPU boost and this laptop does not. **The two
are not the same benchmark and must not be presented as one.** Revision `00001-r47` measured
24.9 s by the same method, so treat **~25–40 s** as the range rather than 39.2 s as a constant.

> ### Measuring a cold start: close the tab first
>
> A Gradio page holds an open `/queue/data` SSE stream, and **Cloud Run counts an open streaming
> request as an active request** — so an instance never retires while any browser tab has the
> demo open. Two 17-minute "idle" windows here produced 1.25 s loads that looked like
> implausibly fast cold starts; they were the same instance, never having scaled down.
>
> This is also a **billing** fact, not just a measurement one: `--min-instances 0` does not mean
> zero cost while someone has the demo open in a background tab. To measure a cold start, close
> every tab pointed at the service, wait ~15 minutes, then load it once — and confirm from the
> logs that the instance start says `AUTOSCALING` rather than `DEPLOYMENT_ROLLOUT`.

Two honest caveats on those numbers:

- **The 62 s excludes the image pull.** Cloud Run's first cold start on a new revision also
  fetches a 2.78 GB image, so its cold start will be meaningfully longer than 62 s. It is
  still far inside the 240 s startup probe with the index baked in — which is the whole
  reason it is baked — but the real figure has to be measured against the deployed service,
  not inferred from here.
- **The generation latency is Ollama on a local CPU**, which is not what Cloud Run will run.
  With the OpenRouter secret bound, generation latency is that provider's, not this one's.

### Size

2.78 GB is dominated by the dependency layer, not by the baked artefacts — CPU-only torch is
confirmed in the image (`torch 2.14.0+cpu`, `torch.version.cuda is None`, no `nvidia-*`
packages). The default CUDA wheel would have added roughly another 2 GB on top of this. The
remaining levers, none taken here because each costs a feature: drop the cross-encoder rerank
checkbox (removes nothing — same `transformers`), drop Gradio for a smaller server, or move
the encoder to an API instead of running it in-process.


---

## Probing the deployed image without handling the key

`OPENROUTER_API_KEY` never enters a local shell or an agent session, which makes "what does the
provider actually return in production?" surprisingly hard to answer. The answer is a **one-off
Cloud Run job on the image digest the service is running**, with the secret bound the same way
the service binds it. The probe runs beside the secret; the secret never travels.

```bash
REG=asia-south1-docker.pkg.dev/production-rag-2026/production-rag/production-rag
DIGEST=$(gcloud run revisions describe production-rag --region asia-south1 \
           --format='value(spec.containers[0].image)')

# The script is passed base64-encoded in an env var, so no rebuild is needed and
# the bootstrap contains no commas -- gcloud splits --args on commas unless you
# give it the ^delim^ prefix.
B64=$(base64 -w0 probe.py)
gcloud run jobs create probe-openrouter \
  --image "$DIGEST" --region asia-south1 \
  --memory 2Gi --cpu 2 --task-timeout 900s --max-retries 0 \
  --set-secrets OPENROUTER_API_KEY=openrouter-api-key:latest \
  --set-env-vars "PROBE_B64=$B64" \
  --command python \
  --args="^@^-c@exec(__import__('base64').b64decode(__import__('os').environ['PROBE_B64']))"

gcloud run jobs execute probe-openrouter --region asia-south1 --wait
gcloud logging read \
  'resource.type="cloud_run_job" AND labels."run.googleapis.com/execution_name"="<execution>"' \
  --limit 200 --format='value(textPayload)' --order asc
gcloud run jobs delete probe-openrouter --region asia-south1   # it holds the secret binding
```

Three rules that made this useful rather than merely clever:

- **Pin the digest, not the tag.** The point is to observe the code that is actually serving.
- **Print the response envelope, never the environment.** The reply cannot contain the key; a
  stray `os.environ` dump can.
- **Budget the calls.** The free tier is 50 requests/day account-wide. The probe that found
  §12.2 cost about 20, and roughly half of those were upstream 502s.

Delete the job when finished. It carries a secret binding and there is no reason to leave one
lying around for a diagnosis that is over.
