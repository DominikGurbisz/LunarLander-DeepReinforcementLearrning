# LunarLander Deep Reinforcement Learning (PyTorch)

Dieses Projekt setzt die Pflichtanforderungen der Aufgabenstellung für **LunarLander** um:
- **DQN** (value-based)
- **REINFORCE** (policy-gradient)

Enthalten sind Training, Evaluation, Logging, Modellspeicherung/-laden, ein separates Demo-Skript und Videoaufzeichnungen pro relevanter Hyperparameter-Einstellung.

## 1) Installation

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## 2) Wichtige Abhängigkeiten

- Python 3.10+
- PyTorch
- Gymnasium + Box2D (`gymnasium[box2d]`)
- NumPy, Pandas, Matplotlib
- imageio + imageio-ffmpeg für Video-Encoding

## 3) Hinweise zu Box2D und Video

### Box2D
Falls `gymnasium[box2d]` bei dir nicht sauber installiert:
1. `pip install --upgrade pip setuptools wheel`
2. Danach erneut `pip install -r requirements.txt`

### Video / ffmpeg
Gymnasium `RecordVideo` nutzt ffmpeg im Hintergrund. Mit `imageio-ffmpeg` klappt es meist ohne Systeminstallation.
Falls keine MP4 erzeugt wird:
1. Prüfe, ob `imageio-ffmpeg` installiert ist.
2. Starte den Lauf erneut mit `--record-video`.
3. Prüfe Schreibrechte im `videos/`-Ordner.

## 4) Projektstruktur

```text
.
├── README.md
├── requirements.txt
├── play_demo.py
├── report_template.md
├── src
│   ├── common
│   │   ├── seed.py
│   │   ├── utils.py
│   │   ├── logger.py
│   │   ├── plotting.py
│   │   └── video.py
│   ├── dqn
│   │   ├── model.py
│   │   ├── replay_buffer.py
│   │   ├── train_dqn.py
│   │   └── evaluate_dqn.py
│   └── reinforce
│       ├── model.py
│       ├── train_reinforce.py
│       └── evaluate_reinforce.py
├── scripts
│   ├── run_all_experiments.py
│   ├── make_report_plots.py
│   └── record_all_videos.py
├── models
├── logs
├── videos
└── results
```

## 5) Gewählte Hyperparameter-Vergleiche

### DQN
- **Algorithmusrelevant:** `epsilon_decay` (low/opt/high)
  - low: `0.990`
  - opt: `0.995`
  - high: `0.999`
- **Frei:** `learning_rate` (low/opt/high)
  - low: `3e-4`
  - opt: `1e-3`
  - high: `3e-3`

### REINFORCE
- **Algorithmusrelevant:** `gamma` (low/opt/high)
  - low: `0.90`
  - opt: `0.99`
  - high: `0.999`
- **Frei:** `hidden_size` (low/opt/high)
  - low: `64`
  - opt: `128`
  - high: `256`

Damit entstehen die 12 Kernkonfigurationen (6 pro Algorithmus).

## 6) Trainingsbefehle

### DQN (einzelner Lauf)
```bash
python -m src.dqn.train_dqn --exp-name dqn_lr_opt_seed1 --seed 1 --lr 0.001 --epsilon-decay 0.995
```

### REINFORCE (einzelner Lauf)
```bash
python -m src.reinforce.train_reinforce --exp-name reinforce_gamma_opt_seed1 --seed 1 --gamma 0.99 --normalize-returns
```

## 7) Alle Kernexperimente starten

```bash
python scripts/run_all_experiments.py --seed 1
```

Optional:
```bash
python scripts/run_all_experiments.py --seed 1 --skip-existing
python scripts/run_all_experiments.py --dry-run
```

## 8) Evaluation

### DQN
```bash
python -m src.dqn.evaluate_dqn --model-path models/dqn/dqn_learning_rate_opt_seed1.pt --episodes 20
```

### REINFORCE
```bash
python -m src.reinforce.evaluate_reinforce --model-path models/reinforce/reinforce_gamma_opt_seed1.pt --episodes 20
```

Mit Video:
```bash
python -m src.dqn.evaluate_dqn --model-path models/dqn/dqn_learning_rate_opt_seed1.pt --record-video --video-folder videos/dqn/eval
python -m src.reinforce.evaluate_reinforce --model-path models/reinforce/reinforce_gamma_opt_seed1.pt --record-video --video-folder videos/reinforce/eval
```

## 9) Separates Demo-Skript (Pflicht)

```bash
python play_demo.py --algo dqn --model-path models/dqn/dqn_learning_rate_opt_seed1.pt --record-video
python play_demo.py --algo reinforce --model-path models/reinforce/reinforce_gamma_opt_seed1.pt --record-video
```

Zusätzlich möglich:
```bash
python play_demo.py --algo dqn --model-path models/dqn/dqn_learning_rate_opt_seed1.pt --render
```

## 10) Videos für alle Hyperparameter-Konfigurationen erzeugen

```bash
python scripts/record_all_videos.py --seed 1
```

Videos werden geordnet gespeichert, z. B.:
- `videos/dqn/epsilon_decay_low/...`
- `videos/reinforce/gamma_high/...`

Dateipräfixe enthalten Konfiguration + Seed, z. B. `dqn_epsilon_decay_low_seed1-episode-0.mp4`.

## 11) Logging und Ergebnisse

- Trainingslogs (CSV + JSON):
  - `logs/dqn/<exp-name>/metrics.csv`
  - `logs/dqn/<exp-name>/summary.json`
  - `logs/reinforce/<exp-name>/metrics.csv`
  - `logs/reinforce/<exp-name>/summary.json`
- Modelle:
  - `models/dqn/*.pt`
  - `models/reinforce/*.pt`
- Evaluationszusammenfassungen:
  - `results/tables/*.json`

Geloggte Größen enthalten u. a.:
- episode, total_reward, moving_avg_reward, episode_length
- loss, learning_rate, seed, exp_name
- DQN: epsilon, buffer_size, mean_q
- REINFORCE: policy_loss, return_mean, return_std, grad_norm

## 12) Plots erzeugen

```bash
python scripts/make_report_plots.py
```

Ausgabe unter `results/plots/`.

## 13) Typische Fehlerquellen

1. **Modellpfad falsch** → `FileNotFoundError` in Eval/Demo.
2. **Hidden Size passt nicht** (Laden mit anderem Netz als Training).
3. **Box2D nicht sauber installiert**.
4. **Video leer/nicht erzeugt** wegen ffmpeg/Codec-Problem.
5. **Zu wenige Episoden** → keine sichtbaren Lerntrends.

## 14) Referenzen (Inspiration)

- CleanRL: https://github.com/vwxyzjn/cleanrl
- GeeksforGeeks REINFORCE: https://www.geeksforgeeks.org/machine-learning/reinforce-algorithm/
- Medium REINFORCE: https://medium.com/@sthanikamsanthosh1994/reinforcement-learning-part-2-policy-gradient-reinforce-using-tensorflow2-a386a11e1dc6

Dieses Projekt ist eigenständig umgesetzt und nicht als 1:1-Kopie der Referenzen.
