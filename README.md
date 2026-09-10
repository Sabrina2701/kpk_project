# KPK world-model project

Emergent vs. supervised representations of outcome-proximity in a
King+Pawn-vs-King move-prediction Transformer. The model is trained only to
predict the next move; the rest of the project checks whether and where
it nonetheless develops an internal representation of how close is the
outcome, using an exact Syzygy tablebase as ground truth, rather than
noisy human games.

## Repository layout

```
kpk_project/
├── tokenizer.py                  move vocabulary, shared across KPK/KRK
├── tablebase.py                  Syzygy wrapper with LRU cache
├── selfplay.py                   position sampling + self-play
├── data_gen.py                   sharded, resumable dataset generation
├── dataset.py                    PyTorch Dataset, M2/M3/M4 labels per ply
├── model.py                      causal Transformer (MoveGPT)
├── probe.py                      linear probes (M2/M3/M4)
├── patch.py                      activation patching (M5)
├── m5_pairs.py                   minimal clean/corrupt pairs for M5
├── train_m1.py                   training loop (M1)
├── validate.py                   6 correctness checks on generated data
├── run_probes.py                 M2 + M3
├── run_m4.py                     M4 (OOD generalization)
├── run_m5.py                     M5 (causal patching)
├── run_baseline_features.py      ablation: geometric features vs. hidden states
├── merge_models.py               M6 (KPK/KRK weight merging)
├── test_tokenizer.py             tokenizer unit tests
├── requirements.txt
└── results/                      run outputs (JSON files)
```

`data/`, `data_krk/`, `checkpoints*/`, `syzygy/` are **not** in this
repository (they are large/binary/generated): they are in Google Drive https://drive.google.com/drive/folders/112j-yVEIZGaiuZJ9I3jE_OmxKkGYSD1S?usp=sharing
It's a public folder.

## Setup

```bash
git clone https://github.com/Sabrina2701/kpk_project.git kpk_project
cd kpk_project
pip install -r requirements.txt --quiet
```

On Colab, then:

```python
from google.colab import drive
drive.mount('/content/drive')
```

The Drive mount stays for the whole session: data, checkpoints and results
are always read/written from there
(`/content/drive/MyDrive/kpk_project/...`), while the code comes from the
clone above — no more uploading zips by hand or patching code at runtime.

### Syzygy tablebases

The 5 K+piece-vs-K tablebases are needed (WDL+DTZ = 10 files, a few tens of
KB each): not just KPvK/KRvK, but also KNvK/KBvK/KQvK, because a promotion
during self-play moves the position into one of those and `probe_dtz()`
needs to be able to query it.

```bash
mkdir -p syzygy
for piece in P N B R Q; do
  wget -q -nc -P syzygy "https://tablebase.lichess.ovh/tables/standard/3-4-5-wdl/K${piece}vK.rtbw"
  wget -q -nc -P syzygy "https://tablebase.lichess.ovh/tables/standard/3-4-5-dtz/K${piece}vK.rtbz"
done
```

## Pipeline (in order)

```bash
# 1) Correctness of generated data
python validate.py --tablebase_dir ./syzygy --n_games 200

# 2) Dataset generation (sharded and resumable: rerunning with a higher
#    n_games picks up where it left off instead of regenerating everything)
python data_gen.py --domain kpk --n_games 200000 \
    --out /content/drive/MyDrive/kpk_project/data --tablebase_dir ./syzygy

# 3) M1: next-move training
python train_m1.py --data_dir /content/drive/MyDrive/kpk_project/data \
    --out_dir /content/drive/MyDrive/kpk_project/checkpoints --epochs 25 --batch_size 512

# 4) M2 + M3: static and outcome probes, per layer
python run_probes.py --data_dir .../data --checkpoint .../checkpoints/best.pt \
    --out results/probe_results.json

# 5) M4: near->far generalization on |DTZ|
python run_m4.py --data_dir .../data --checkpoint .../checkpoints/best.pt \
    --out results/m4_results.json

# 6) M5: causal patching (isolating a layer's own marginal contribution)
python run_m5.py --data_dir .../data --checkpoint .../checkpoints/best.pt \
    --tablebase_dir ./syzygy --out results/m5_results_vanilla_v2.json


# 7) Ablations: multitask (auxiliary WDL/DTZ heads) and geometric baseline
python train_m1.py ... --out_dir .../checkpoints_multitask --multitask
python run_baseline_features.py --data_dir .../data --n_games 10000 \
    --out results/baseline_features_results_v2.json

# 8) M6: KRK specialist fine-tuned from the KPK checkpoint, then merge
python data_gen.py --domain krk --n_games 50000 --out .../data_krk --tablebase_dir ./syzygy
python train_m1.py --data_dir .../data_krk --out_dir .../checkpoints_krk \
    --domain krk --epochs 15 --batch_size 512 --init_from .../checkpoints/best.pt
python merge_models.py --base_checkpoint .../checkpoints/best.pt \
    --finetuned_checkpoint .../checkpoints_krk/best.pt \
    --data_dir_kpk .../data --data_dir_krk .../data_krk --out results/merge_results.json
```

Repeating steps 4-6 on both the vanilla and the multitask checkpoint
produces the pairs of files used in the report's comparisons
(`*_vanilla_*` / `*_multitask_*`), and repeating with a different `--seed`
or a different `--near_dtz_max` produces the stability replications cited
in the Appendix.

## Two ways to check the results

- **From scratch**: follow the full pipeline above (needs a GPU and time,
  especially for the 200k self-play games and the 25 training epochs).
- **Without regenerating**: the JSON files in `results/` are already in
  this repository; the checkpoints
  (`checkpoints/best.pt`, `checkpoints_multitask/best.pt`,
  `checkpoints_krk/best.pt`) are on Drive instead, not in the repository. You can find
  the link in the section above.

## Design decisions and known criticalities, and where they are addressed

- **Shared KPK/KRK vocabulary.** `tokenizer.py`: built from pure move
  geometry, 1859 tokens (including bishop/knight geometry, needed only for
  promotions), identical by construction across the two domains, a
  precondition for weight merging in M6.
- **"Non-Markovian."** `tokenizer.py`, docstring: the model's input is the
  move sequence, not the FEN at every ply — M2 tests state reconstruction,
  M3/M4/M5 additionally test for a signal that requires look-ahead.
- **Self-play too easy.** `selfplay.py`: `p_critical_start=0.75` and
  `critical_king_radius=1`, tuned empirically (radius=1 nearly doubles the
  true zugzwang rate compared to radius=2/3); verified on real data: ~19%
  zugzwang for generic positions, ~57% for critical ones, reproduced across
  two independent seeds.
- **Cost of tablebase probing at scale.** `tablebase.py`: LRU cache keyed
  on FEN (stripped of halfmove/fullmove clocks).
- **M5 needs to show a real causal effect, not just correlation.**
  `patch.py`: isolates a single layer's own marginal contribution (not the
  full accumulated state), since a full overwrite trivially recovers 100%
  by construction on single-token-difference pairs — see the module
  docstring.





