# LLM Reasoning with MCTS

Enhancing large language model (LLM) reasoning through **Monte Carlo Tree Search (MCTS)**. Rather than sampling a single chain of thought, this framework builds an explicit search tree over partial reasoning steps, uses PUCT-based selection to focus compute on promising branches, and iteratively fine-tunes a local LLaMA2 model on high-reward trajectories — progressively reducing reliance on GPT-4.

A Tree-of-Thoughts (BFS) baseline is included for comparison on the same benchmarks.

---

## How It Works

```
┌─────────────────────────────────────────────────────────┐
│                     MCTS + Fine-tuning Loop             │
│                                                         │
│  for each training iteration:                           │
│    for each puzzle:                                     │
│      ┌── Selection  (PUCT)                              │
│      │   traverse existing nodes to find a leaf         │
│      ├── Expansion                                      │
│      │   generate K candidate next steps                │
│      │   via GPT-4 (early) or LLaMA2 (later)           │
│      ├── Simulation                                     │
│      │   roll each candidate to completion              │
│      │   compute letter-level reward                    │
│      └── Backpropagation                                │
│          update Q and N along the path                  │
│                                                         │
│    Fine-tune LLaMA2 (LoRA) on best trajectories        │
│    Evaluate on held-out validation set                  │
└─────────────────────────────────────────────────────────┘
```

**GPT-4 → LLaMA2 annealing:** GPT-4 drives expansion in early iterations (90% probability), linearly handing off to the locally fine-tuned LLaMA2 over the course of training. This lets the model bootstrap from a strong prior and then self-improve.

---

## Repository Structure

```
.
├── src/
│   ├── mcts/
│   │   ├── crossword_mcts.py   # CrosswordsEnv: state, reward, prompt construction
│   │   └── mcts_cot.py         # MCTS search, rollout, LLaMA2 fine-tuning loop
│   └── tot/
│       ├── models.py            # OpenAI ChatCompletion wrapper with retry/backoff
│       ├── methods/bfs.py       # BFS solver (Tree-of-Thoughts baseline)
│       ├── prompts/             # Few-shot prompts for each task
│       ├── tasks/               # Task definitions (crosswords, game24, text)
│       └── data/                # Benchmark datasets
├── scripts/                     # Shell scripts for reproducing experiments
├── run.py                       # Entry point for the ToT baseline
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

Set environment variables:

```bash
export OPENAI_API_KEY=<your_openai_key>
export HUGGING_FACE_HUB_TOKEN=<your_hf_token>   # required for LLaMA2
```

---

## Usage

### MCTS + LLaMA2 (mini crosswords)

```bash
cd src/mcts
python mcts_cot.py \
    --model_name meta-llama/Llama-2-13b-hf \
    --use_peft \
    --load_in_8bit \
    --batch_size 4 \
    --gradient_accumulation_steps 2 \
    --num_train_epochs 10 \
    --learning_rate 2e-5
```

Generated trajectories are saved under `saved_generation/crosswords/`.  
Final rewards per iteration are pickled to `saved_generation/crosswords/result.pkl`.

### Tree-of-Thoughts baseline (BFS)

**Game of 24:**
```bash
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
```

**Mini crosswords (CoT sampling):**
```bash
python run.py \
    --task crosswords \
    --backend gpt-4 \
    --naive_run \
    --prompt_sample cot \
    --n_generate_sample 10 \
    --task_start_index 0 \
    --task_end_index 20
```

See `scripts/` for the full set of experiment configurations.

---

## Key Hyperparameters

| Parameter | Default | Description |
|---|---|---|
| `rollout_num` | 10 | MCTS rollouts per puzzle step |
| `depth_limit` | 20 | Maximum reasoning depth |
| `num_different_action` | 5 | Candidate actions generated per expansion |
| `train_iterations` | 20 | Fine-tuning iterations |
| `data_num_per_training` | 10 | Puzzles sampled per iteration |
| PUCT `c` | 1.0 | Exploration constant |

---

## Requirements

- Python ≥ 3.8
- CUDA GPU (strongly recommended for LLaMA2 fine-tuning; 24 GB+ VRAM for 13B in 8-bit)
- OpenAI API key with GPT-4 access
- Hugging Face token with LLaMA2 access

Full dependency list: [`requirements.txt`](requirements.txt)

---

## License

[MIT License](LICENSE)
