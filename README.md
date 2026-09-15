<!--
README skeleton. Sections are ordered per the build spec (section 10).
Fill in as stages complete; the eval results table is the credibility anchor.
-->

# RAG Knowledge Agent

> Reference implementation built as a portfolio project. Not actively maintained
> for external contributions, but issues/questions are welcome.

<!-- TODO(attribution): link to site / LinkedIn / agency page here (top). -->

## The problem

<!-- TODO(stage 11): one-paragraph business problem statement from the spec.
Companies accumulate scattered internal knowledge (policies, manuals, wikis,
support FAQs, compliance docs). People waste time searching; new hires take weeks
to get self-sufficient. Generic chatbots make it worse by answering confidently
from general knowledge instead of the company's actual docs. -->

## Naive RAG vs. this

<!-- TODO(stage 3/11): concrete side-by-side. Naive RAG confidently answers a
question with no supporting document; this system returns a grounded refusal:
"I don't have information on that in the provided documents." -->

## Architecture

<!-- TODO(stage 11): embed the diagram. Source: docs/architecture.md -->

See [docs/architecture.md](docs/architecture.md).

## Eval results

<!-- TODO(stage 8/11): prominent table.
| Metric | Naive RAG | This system |
|---|---|---|
| Faithfulness | ... | ... |
| Answer relevancy | ... | ... |
| Context precision | ... | ... |
| Context recall | ... | ... |
Measured on a 30-50 question golden set including no-answer and adversarial cases. -->

## Quickstart

```bash
cp .env.example .env
# set RAG_CORE_SHARED_SECRET (openssl rand -hex 32)
make up && make health
```

Without `make` (e.g. Windows PowerShell):

```powershell
Copy-Item .env.example .env
# set RAG_CORE_SHARED_SECRET
.\scripts\dev.ps1 up
.\scripts\dev.ps1 health
```

<!-- TODO(stage 11): full quickstart - fetch sample corpus, ingest, ask a
question. Target: under 5 minutes from clone to first answer. -->

## What I'd change for a production client deployment

<!-- TODO(stage 11): honest demo-scoped vs. production-hardened section -
auth, multi-tenancy, rate limiting, monitoring, secrets management. -->

## Known limitations

<!-- TODO(stage 11): stated upfront. Multi-hop questions, no live source
connectors, no enforced multi-tenancy, no end-user auth, no agentic actions. -->

## License

[Apache-2.0](LICENSE). See [NOTICE](NOTICE) for corpus attribution.

<!-- TODO(attribution): link to site / LinkedIn / agency page here (bottom). -->
