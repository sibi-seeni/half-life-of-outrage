# Running the pipeline

Everything runs on your Mac except the RoBERTa baseline, which runs as a GPU job on HiPerGator.
The study window ends on **2026-04-30** (`END_DATE` in `src/hlo/config.py`). Collection skips videos published and comments posted after that date, and `build_dataset` drops any later rows already in the database.

## Part 1: on the Mac

```bash
cd half-life-of-outrage
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt && pip install -e .
cp .env.example .env            # fill in YOUTUBE_API_KEY and NAVIGATOR_API_KEYS
pytest                          # 22 passed, no data needed
```

Put your existing files in `data/`:
- the database, renamed to `data/arc_raiders.db`
- `comments_for_analysis.csv` renamed to `comments_clean.csv`
- `comments_llama_labeled.csv` renamed to `labels_llm.csv`

Then, in order:

```bash
python -m hlo.collect full          # adds 2026-03-14 .. 2026-04-30; resumable, see quota note below
python -m hlo.data.build_dataset    # rebuilds data/comments_clean.csv (up to 2026-04-30)
python -m hlo.label.llm_labeler --limit 5     # smoke test: check data/labels_llm.csv looks right
python -m hlo.label.llm_labeler     # labels only comments not labeled yet (resumable, Ctrl-C safe)
```

**YouTube quota.** A search costs 100 of the 10,000 daily units, so discovery needs roughly 2–4 days of runs. Per-minute limits are handled automatically (it pauses and retries). When the daily quota runs out it prints "Daily YouTube quota used up", saves progress and exits; run the same command the next day (quota resets at midnight Pacific). Finished search windows are skipped on reruns. Once discovery reports nothing new, `python -m hlo.collect comments` alone fetches comments (1 unit per page, much cheaper).

`caffeinate -i` in front of a long command stops the Mac from sleeping, e.g. `caffeinate -i python -m hlo.label.llm_labeler`.
Labeling goes about 4 s per comment, so a large batch takes hours. It resumes where it left off.

Then send `data/comments_clean.csv` to HiPerGator (Part 2). When `labels_roberta.csv` comes back, finish on the Mac:

```bash
python -m hlo.eval.gold export      # annotation sheet for you + annotator 2
./scripts/run_all.sh                # analysis + robustness grid -> results/, figures/
```

## Part 2: RoBERTa on HiPerGator (SLURM)

### How SLURM works, in short
When you log in to HiPerGator you land on a **login node**. It's shared by everyone, so it's fine for copying files, editing and submitting jobs, but not for heavy computing. Heavy work goes in a **job script**: a shell script whose `#SBATCH` lines say what you need (account, GPU, memory, time). `sbatch` puts it in the queue, SLURM starts it on a compute node when resources are free, and everything it prints goes to a log file. You can log out while it runs.

The `#SBATCH` lines in `scripts/hipergator/*.sbatch` already match the settings your professor gave: account and QOS `eee6778`, partition `hpg-b200`, `--gres=gpu:b200:1`, 3 hours.

### Step 1: log in and get the code (once)
```bash
ssh <gatorlink>@hpg.rc.ufl.edu                         # password + Duo
cd /blue/eee6778/$USER 2>/dev/null || cd ~             # group storage if it exists, else home
git clone https://github.com/sibi-seeni/half-life-of-outrage.git
cd half-life-of-outrage
```

### Step 2: build the environment (once, about 15 minutes)
```bash
sbatch scripts/hipergator/setup_env.sbatch             # prints: Submitted batch job 1234567
squeue -u $USER                                        # ST column: PD = waiting, R = running; gone = finished
tail -f logs/hlo-setup_1234567.out                     # watch the log live (Ctrl-C stops watching, not the job)
```
It should end with `Setup finished.` This installs into `/blue/eee6778/$USER/hlo`, not into the system, so you don't need admin rights. If `/blue/eee6778` doesn't exist for you, run `export HLO_WORK=$HOME/hlo` before both `sbatch` commands.

### Step 3: copy the dataset up (from a terminal on the Mac)
```bash
scp data/comments_clean.csv <gatorlink>@hpg.rc.ufl.edu:/blue/eee6778/<gatorlink>/half-life-of-outrage/data/
```
(Use `~/half-life-of-outrage/data/` instead if you cloned into your home folder.)

### Step 4: run RoBERTa
```bash
cd /blue/eee6778/$USER/half-life-of-outrage
sbatch scripts/hipergator/roberta.sbatch
squeue -u $USER
tail -f logs/hlo-roberta_<jobid>.out
```
The log first prints the GPU name (should say B200), then progress. ~200k comments should take minutes on a B200, not hours.

### Step 5: copy the result back (on the Mac)
```bash
scp <gatorlink>@hpg.rc.ufl.edu:/blue/eee6778/<gatorlink>/half-life-of-outrage/data/labels_roberta.csv data/
```

### Useful commands
| Command | What it does |
|---|---|
| `squeue -u $USER` | your queued/running jobs |
| `scancel <jobid>` | cancel a job |
| `sacct -j <jobid> --format=JobID,State,Elapsed,MaxRSS` | how a finished job went (COMPLETED / FAILED / TIMEOUT) |
| `cat logs/<name>_<jobid>.out` | full output of a job |

### If something goes wrong
- **Job sits in `PD` for a long time:** the B200s are busy. Leave it queued. `squeue -u $USER --start` shows an estimated start time.
- **`no GPU visible` or a CUDA error:** check the log's `nvidia-smi` line, and send me the last 30 lines of the log.
- **`pip` or model download fails in the setup job:** the compute node may not have internet. Run the same `pip install` lines on the login node instead (installing into your own folder is allowed there), then resubmit.
- **`TIMEOUT`:** raise `#SBATCH --time` in `roberta.sbatch` (interactive GPU sessions are capped at 12 h; batch jobs can usually go longer).
