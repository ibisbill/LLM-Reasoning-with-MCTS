# LLM Reasoning with MCTS

This repository explores enhancing LLM reasoning through **Monte Carlo Tree Search (MCTS)**. The MCTS framework guides the generation process by iteratively building a search tree, evaluating partial solutions, and fine-tuning a local LLaMA2 model on high-reward trajectories.

The Tree-of-Thoughts (ToT) baseline is included for comparison on the same task benchmarks.

![Teaser](pics/teaser.png)

---

## Overview

### MCTS + LLM (`src/mcts/`)

The core contribution. At each reasoning step:

1. **Selection** — traverse existing tree nodes using PUCT scoring
2. **Expansion** — generate candidate next steps via GPT-4 or LLaMA2
3. **Simulation** — roll each candidate out to completion and compute a reward
4. **Backpropagation** — update Q/N values along the visited path
5. **Fine-tuning** — periodically fine-tune LLaMA2 on high-reward trajectories (LoRA / QLoRA)

### Tree of Thoughts baseline (`src/tot/`)

BFS-based deliberate reasoning over candidate thoughts, as described in [Yao et al., 2023](https://arxiv.org/abs/2305.10601). Supports Game of 24, mini crosswords, and creative writing tasks.

---

## Repository Structure

```
.
├── src/
│   ├── mcts/
│   │   ├── crossword_mcts.py   # CrosswordsEnv and MiniCrosswordsTask
│   │   └── mcts_cot.py         # MCTS construction, rollout, and LLaMA2 fine-tuning loop
│   └── tot/
│       ├── __init__.py
│       ├── models.py            # OpenAI API wrapper
│       ├── methods/
│       │   └── bfs.py           # BFS solver for ToT
│       ├── prompts/             # Task-specific prompts
│       │   ├── crosswords.py
│       │   ├── game24.py
│       │   └── text.py
│       ├── tasks/               # Task definitions
│       │   ├── base.py
│       │   ├── crosswords.py
│       │   ├── game24.py
│       │   └── text.py
│       └── data/                # Benchmark datasets
│           ├── 24/
│           ├── crosswords/
│           └── text/
├── scripts/                     # Shell scripts for running experiments
│   ├── crosswords/
│   ├── game24/
│   └── text/
├── logs/                        # Pre-computed experiment results
├── run.py                       # ToT entry point
├── requirements.txt
└── setup.py
```

---

## Installation

```bash
git clone https://github.com/ibisbill/LLM_Reasoning_w_MCTS.git
cd LLM_Reasoning_w_MCTS
pip install -e .
```

Set your OpenAI API key:

```bash
export OPENAI_API_KEY=your_key_here
```

---

## Usage

### MCTS + LLaMA2 (crosswords)

```bash
cd src/mcts
python mcts_cot.py \
    --model_name meta-llama/Llama-2-13b-hf \
    --use_peft \
    --batch_size 4 \
    --gradient_accumulation_steps 2 \
    --load_in_8bit \
    --num_train_epochs 10 \
    --learning_rate 2.0e-5
```

The script will:
- Alternate between GPT-4 (for early exploration) and LLaMA2 (as training progresses) to generate candidate actions
- Fine-tune LLaMA2 via LoRA after each iteration
- Save generated trajectories under `saved_generation/crosswords/`

### Tree of Thoughts (BFS)

```bash
# Game of 24
python run.py \
    --task game24 \
    --backend gpt-4 \
    --method_generate propose \
    --method_evaluate value \
    --method_select greedy \
    --n_generate_sample 1 \
    --n_evaluate_sample 3 \
    --n_select_sample 5 \
    --task_start_index 900 \
    --task_end_index 1000

# Mini crosswords (naive CoT)
python run.py \
    --task crosswords \
    --backend gpt-4 \
    --naive_run \
    --prompt_sample cot \
    --n_generate_sample 10
```

See `scripts/` for more experiment configurations.

---

## Key Design Choices

| Component | Detail |
|---|---|
| **Action generation** | GPT-4 (Azure) or LLaMA2 beam search |
| **Selection policy** | PUCT with exploration constant c=1.0 |
| **Reward** | Letter-level accuracy on the 5×5 grid |
| **Fine-tuning** | SFTTrainer + LoRA (PEFT) |
| **GPT → LLaMA2 annealing** | GPT usage decays linearly from 90% → 10% over training |

---

## Requirements

- Python ≥ 3.8
- CUDA GPU (recommended for LLaMA2 fine-tuning)
- OpenAI API key (GPT-4 access)
- Hugging Face token (LLaMA2 access)

See `requirements.txt` for the full dependency list.

---

## License

MIT License. See [LICENSE](LICENSE).
