# Reproducing Magentic Marketplace across LLM providers

This folder reproduces the main experiments of
**Magentic Marketplace: An Open-Source Environment for Studying Agentic Markets**
(Bansal et al., Microsoft Research, 2025, [arXiv:2510.25779](https://arxiv.org/abs/2510.25779))
on four kinds of model: OpenAI, Anthropic, Google Gemini and self-hosted open-weight models served with vLLM.

All simulation, agent and analytics code is Microsoft's, from
[microsoft/multi-agent-marketplace](https://github.com/microsoft/multi-agent-marketplace) (MIT licence).
This folder adds a single experiment runner, a model registry and a results summary on top of it.
It is the first phase of a study of LLM seller-agent collusion in energy markets; the extension will live in a separate folder.

## What gets reproduced

| Experiment | Question | Data | Conditions | Paper finding to check |
| --- | --- | --- | --- | --- |
| `welfare` | How much welfare do LLM agents create for customers? | `mexican_33_99`, `contractors_33_99` | lexical (realistic) vs optimal (perfect) search | Frontier models beat simple baselines with lexical search and get close to optimal welfare with perfect search; Qwen3-14B does poorly. |
| `search` | Does showing more search results help? | `mexican_100_300` | 3, 10, 30, 100 results shown | More results lowered welfare (a "paradox of choice"); the size of the drop varied widely by model. |
| `manipulation` | Can sellers manipulate buyer agents? | `malicious_description/*` | control, authority, social proof, loss aversion, basic and strong prompt injection | Some frontier models resisted most tactics; GPT-4o, GPT-OSS-20B and Qwen3-4B were vulnerable, and prompt injection often redirected payments. |
| `position` | Do agents favour the first result or first proposal? | `position_bias/*` | target seller listed or proposing 1st, 2nd, 3rd | Frontier models chose search positions almost uniformly, but every model strongly favoured the first proposal received. |

Paper findings are summarised from the [arXiv paper](https://arxiv.org/html/2510.25779v1); see it for exact numbers.
Two settings here are my reading of the paper rather than taken from its code: the welfare experiment's dataset sizes
and treating the `optimal` search algorithm as "perfect search". Each upstream runner's settings
(search algorithm, 10 results, 100 customer steps, 5 runs) are kept as they are.

## Setup (macOS or Linux)

You need [uv](https://docs.astral.sh/uv/), [Docker Desktop](https://www.docker.com/products/docker-desktop/) and API keys for the providers you use.

```bash
git clone https://github.com/<your-username>/multi-agent-marketplace.git
cd multi-agent-marketplace
uv sync --all-extras
cp sample.env .env          # then add your keys (see below)
docker compose up -d        # starts Postgres on localhost:5432
```

Add to `.env`:

```bash
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."
export GEMINI_API_KEY="..."

# Only for self-hosted models
export VLLM_BASE_URL="http://localhost:8001/v1"
export VLLM_API_KEY="dummy"
```

To run Qwen3-4B on a laptop without a GPU server, install [Ollama](https://ollama.com), run `ollama pull qwen3:4b`,
and use `--models qwen3-4b-ollama`. It is free but slow. For the exact model the paper used (no thinking step, faster), run
`ollama pull hf.co/unsloth/Qwen3-4B-Instruct-2507-GGUF:Q4_K_M` and use `--models qwen3-4b-instruct-ollama`.

For self-hosted models, start vLLM first, for example `vllm serve Qwen/Qwen3-14B --port 8001`.
Model names and endpoints live in [`models.yaml`](models.yaml); add a model there to use it.

## Running

Always start with the smoke test. It runs 3 customers and 9 restaurants once with the cheapest model and checks the whole pipeline:

```bash
uv run repro/run_repro.py --smoke
```

Then run experiments. `--dry-run` prints the plan and the number of runs without calling any model:

```bash
uv run repro/run_repro.py --experiments welfare --models gpt-4o --runs 1 --dry-run
uv run repro/run_repro.py --experiments welfare --models gpt-4o qwen3-14b --runs 5
uv run repro/run_repro.py --experiments all --models gpt-4.1 claude-sonnet-4-5 gemini-2.5-flash
uv run repro/run_repro.py --experiments manipulation --models gpt-4o --conditions prompt_injection
```

For cheaper development runs, `welfare-small` repeats the welfare experiment on the
10-customer, 30-business markets; it is not one of the paper's settings.

Runs are resumable: a finished run is skipped, and a failed run is retried next time.
The full grid is 26 conditions: 130 runs per model at 5 runs each. Check the cost of a 1-run pass before running everything.

## Results

```bash
uv run repro/summarize.py          # welfare table across runs
uv run repro/view_run.py           # newest run as readable conversations
uv run repro/view_run.py path/to/run.db --customer customer_0001
```

`view_run.py` prints each customer's goal, searches, messages, proposals and payments in order,
then how the run ended for them, scored the same way as the upstream analytics. It also saves
the transcript as `<run>.transcript.md` next to the database.

* `results/summary.csv`: one row per run (customer welfare, needs met, LLM calls, failures).
* `results/summary.md`: mean ± standard deviation per model and condition.
* `paper_experiments/<experiment>/results/`: one SQLite database and analytics JSON per run, named the way the upstream analysis scripts expect.

For the paper's experiment-specific metrics, run the upstream scripts on the same databases:

```bash
cd experiments/malicious && python analyze_results.py && python plot_results.py
cd experiments/position  && python generate_position_data.py && python generate_proposal_data.py \
                         && python plot_position_bias.py && python plot_proposal_bias.py
```

## Reproduction status

| Experiment | Models run | Matches paper? | Notes |
| --- | --- | --- | --- |
| welfare | – | – | |
| search | – | – | |
| manipulation | – | – | |
| position | – | – | |

## Citation

If you use this, please cite the original paper:

```bibtex
@misc{bansal-arxiv-2025,
  title         = {Magentic Marketplace: An Open-Source Environment for Studying Agentic Markets},
  author        = {Gagan Bansal and Wenyue Hua and Zezhou Huang and Adam Fourney and Amanda Swearngin and Will Epperson and Tyler Payne and Jake M. Hofman and Brendan Lucier and Chinmay Singh and Markus Mobius and Akshay Nambi and Archana Yadav and Kevin Gao and David M. Rothschild and Aleksandrs Slivkins and Daniel G. Goldstein and Hussein Mozannar and Nicole Immorlica and Maya Murad and Matthew Vogel and Subbarao Kambhampati and Eric Horvitz and Saleema Amershi},
  year          = {2025},
  eprint        = {2510.25779},
  archivePrefix = {arXiv},
  primaryClass  = {cs.MA},
  url           = {https://arxiv.org/abs/2510.25779}
}
```
