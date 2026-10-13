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


---

## Tracing (optional)

> **Status, stated plainly.** The live Cloud Run service is **unchanged and untraced**. The code
> to trace it exists and is switched on by environment variables alone, but nothing is set, and
> **live tracing is deferred pending a hosting and cost decision that is the owner's to make**
> (below). Until then the deployed image behaves exactly as it did before tracing existed.

What it gives: one trace per question, with a span for each stage (`rewrite`, `retrieve.bm25`,
`retrieve.dense`, `fuse`, `rerank`, `generate`), timings, token counts and a cost per model call.
The design, the span names and the attribute names are in `src/production_rag/tracing.py`; the
overhead budget it is held to is registered in `NOTES.md`.

### What a trace is

The served path is the Gradio handler: each request runs its retrieval and generation in one thread and
produces one `rag.query` trace with the full span tree. The batch commands behave differently. `eval` runs
its retrievals one after another and each is its own `rag.query` trace, but `answer-eval` and the other
commands that call the model from a thread pool produce **one trace per LLM call** (a lone `generate` or
`rewrite` span) rather than a tree, because the worker threads do not inherit a root. Count traces
accordingly if those commands are ever run with tracing on.

A response served from the replay cache keeps its recorded token counts but is exported with a cost of 0,
flagged `production_rag.cached=true`; what the original call cost is kept in `production_rag.recorded_cost_usd`.
Summing `cost_usd` over a period therefore gives spend, and summing `recorded_cost_usd` gives what the same
answers would have cost uncached.

### What is and is not exported

| Exported always | Exported only with `PRODUCTION_RAG_TRACE_CONTENT=1` | Never exported |
|---|---|---|
| stage names and durations, retrieval arm, `k`, strategy, model and provider, token counts, cost, whether the answer was cached or refused, the error *type* of a failed stage | the question text, the prompt sent to the model (which contains the retrieved passages) and the model's reply | stack traces, API keys, request headers, the visitor's IP address |

The question a visitor types is **user data**, and so is anything that quotes it, so content capture
is **off by default** and should stay off on the public demo. The retrieved passages come from public
documentation, but they sit in the same prompt as the question. The metadata that is exported is
low-risk but not nothing: it shows when someone used the demo and whether an answer was refused. Error
*messages* are reduced to a fixed prefix (for a provider failure, `HTTP 403`, `could not reach`, `replay mode`
and a few others): the response body and URL a provider error carries are never exported, because a body can
echo the prompt. For any other exception only its type is kept. With content capture on, the full text is kept.

Where trace data lives and for how long is decided by the backend, not by this repository:
Langfuse Cloud keeps it for the plan's data-access window; a self-hosted Langfuse or Phoenix keeps it
until someone deletes it (no retention policy is configured in `tracing.compose.yml`). If content
capture is ever turned on for a public service, say so on the page.

### Switching it on (nothing here is set today)

Tracing needs the `tracing` extra in the image, which the current `space/Dockerfile` does **not**
install (it was left alone so the deployed image is unchanged and its pinned stack is not disturbed
by an unbuilt change). The one-line change, when the owner decides:

```diff
-        ".[embed]" \
+        ".[embed,tracing]" \
```

Without the extra, setting the variables below logs one warning and the app runs untraced.

| Variable | Value | Notes |
|---|---|---|
| `OTEL_EXPORTER_OTLP_ENDPOINT` | the collector's base URL | Setting this is what turns tracing on. `PRODUCTION_RAG_TRACING=0` forces it off even when this is set; `PRODUCTION_RAG_TRACING=1` turns it on against the default `localhost:4318`. |
| `OTEL_EXPORTER_OTLP_HEADERS` | `Authorization=Basic <base64(public:secret)>` for Langfuse | Carries a credential: bind it from **Secret Manager**, never `--set-env-vars`. |
| `PRODUCTION_RAG_TRACE_CONTENT` | unset | Leave unset. See above. |
| `PRODUCTION_RAG_TRACE_EXPORT_TIMEOUT_S` | default `5` | Upper bound on one export attempt. Export runs on a background thread, so a dead collector costs dropped spans and a log line, not request latency. |

Endpoints, both OTLP over HTTP (gRPC is not used):

| Backend | `OTEL_EXPORTER_OTLP_ENDPOINT` | Auth |
|---|---|---|
| Langfuse (self-hosted v3.22 or later) | `https://<host>/api/public/otel` | Basic, from the project's public and secret key |
| Langfuse Cloud (EU) | `https://cloud.langfuse.com/api/public/otel` (US: `https://us.cloud.langfuse.com/api/public/otel`) | same; also add `x-langfuse-ingestion-version=4` to the headers |
| Arize Phoenix | `http://<host>:6006` (traces go to `/v1/traces`) | none by default |

Source for the Langfuse endpoints, headers and attribute mapping:
<https://langfuse.com/integrations/native/opentelemetry> (read 2026-10-13).

```bash
# Store the header value once. It is "Authorization=Basic " followed by base64("<public>:<secret>").
gcloud secrets create otel-headers --replication-policy=automatic --data-file=-

gcloud run services update production-rag --region "$REGION" \
  --set-env-vars OTEL_EXPORTER_OTLP_ENDPOINT=https://<collector>/api/public/otel \
  --set-secrets OTEL_EXPORTER_OTLP_HEADERS=otel-headers:latest
```

**Cloud Run caveat, untested here.** By default Cloud Run throttles an instance's CPU between
requests, and the exporter sends batches from a background thread on a timer. Spans can therefore be
delayed until the next request, and an instance that is stopped with spans still queued loses them.
The remedies are `--no-cpu-throttling` (instance-based billing, which costs more) or a short
`OTEL_BSP_SCHEDULE_DELAY` (milliseconds) so batches leave sooner; neither has been tried on the live
service. Treat the first week of live traces as incomplete.

Shutdown adds a second limit. The exporter's own timeout is 5 seconds per attempt
(`PRODUCTION_RAG_TRACE_EXPORT_TIMEOUT_S`; the batch processor's separate export-timeout setting is not
read by the SDK, so it is not what bounds this). If the collector accepts connections and then drops
them, the flush at shutdown can run for up to about 30 seconds, while Cloud Run stops an instance about
10 seconds after SIGTERM. Spans still queued at that moment are lost. Neither number has been measured
against the live service.

### Hosting options and what they cost

Every figure is an **estimate** from the source and date beside it. Prices change; confirm on the
vendor's page before deciding. None of this is spent or committed.

| Option | Estimated monthly cost | What it needs | Honest trade-off |
|---|---|---|---|
| **Langfuse Cloud, Hobby** | $0 (free, no card). 50k "units" a month, 30 days of data, 2 users. *Source: langfuse.com/pricing, 2026-10-13.* | The owner's Langfuse account and project keys. | The cheapest and quickest. A unit is any trace, span or score, so one answered question is roughly 6 units on the served arm (root, `retrieve.bm25`, `retrieve.dense`, `fuse`, `generate`, plus the trace), which is **on the order of 8,000 questions a month** before the cap. Traces go to a third party's cloud. Past the cap, ingestion stops or the next plan applies: Core is $29/month for 100k units and 90 days (same source). |
| **Self-hosted Langfuse v3** on a small VM | About **$49/month** for a 2 vCPU / 8 GB VM (GCP `e2-standard-2`, us-central1, on demand, about $0.067/hour; third-party price trackers, <https://calculator.holori.com/gcp/vm/e2-standard-2> and <https://gcloud-compute.com/e2-standard-2.html>, one of them dated January 2026; **not verified against Google's own price list**), **plus** disk, plus the owner's time to patch and back up it. The VM size is an estimate: Langfuse recommends at least 2 CPUs and 4 GB for its own containers alone (langfuse.com self-hosting docs, v4 page, 2026-10-13), and ClickHouse, Postgres and Redis run beside them. | A VM with Docker, a stable address and TLS in front, `tracing.compose.yml` adapted for non-loopback access. | Full control of where the data lives and how long it is kept. It is the most expensive and the most work, and it would cost more than the demo itself does. This repository's compose file is for local verification and binds to loopback; it is not a production deployment. |
| **Arize Phoenix** on a small VM, with a persistent store | Roughly **$10 to $15/month** for a very small VM (an estimate, not a quote: the figure above scaled down for a smaller machine), more if a managed Postgres is used. | One container, `arizephoenix/phoenix`, plus a disk or a database. | Lightest to run. **Cloud Run cannot host it usefully**: its filesystem is ephemeral, so traces would vanish on restart unless a persistent database is attached, at which point it is a VM again. |
| **Arize AX Free** (hosted) | $0: 25k spans a month, 15 days of data. *Source: arize.com/pricing, 2026-10-13.* | An Arize account. | Smaller allowance than Langfuse Hobby (about 25k spans, so roughly 5,000 questions a month at 5 spans each) and a short window. |

The honest default, if the owner wants live traces at no cost, is Langfuse Cloud Hobby with content
capture off. The honest alternative, if the owner does not want trace metadata to leave their
infrastructure, is a VM, and that is a recurring cost the demo has not had until now. **No choice has
been made.**

### Verifying it locally first

`tracing.compose.yml` runs a self-hosted Langfuse on loopback with a headless first-run setup. It
has not been started as part of building this. See the header of that file and the "Tracing"
subsection of the README for the commands.
