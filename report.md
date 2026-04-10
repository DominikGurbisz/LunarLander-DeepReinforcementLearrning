# Report: LunarLander-v3 – Vergleich von DQN und REINFORCE

## 1) Zielsetzung
In diesem Bericht werden die auf dem Branch vorhandenen Trainingsplots, Evaluationsmetriken und Videos systematisch ausgewertet, um **DQN** (value-based) und **REINFORCE** (policy-gradient) auf `LunarLander-v3` zu vergleichen.

## 2) Setup und Hyperparameter
Die Experimente folgen dem in `README.md` dokumentierten Raster mit je zwei Hyperparameter-Studien pro Algorithmus (low/opt/high):

- **DQN**
  - `epsilon_decay`: 0.990 / 0.995 / 0.999
  - `learning_rate`: 3e-4 / 1e-3 / 3e-3
- **REINFORCE**
  - `gamma`: 0.90 / 0.99 / 0.999
  - `hidden_size`: 64 / 128 / 256

Pro Konfiguration liegen Trainingslogs und Plots vor; zusätzlich wurden Evaluationen mit 10 Episoden (Seed 100) für die geforderten Vergleichsachsen gespeichert.

## 3) Quantitative Kernergebnisse

### 3.1 Evaluation (10 Episoden, Seed 100)

| Vergleich | Konfiguration | Mean Reward | Std | Min | Max |
|---|---|---:|---:|---:|---:|
| DQN `epsilon_decay` | low | **194.67** | 100.61 | 33.83 | 295.98 |
| DQN `epsilon_decay` | opt | 169.78 | 93.24 | 37.81 | **299.10** |
| DQN `epsilon_decay` | high | -39.88 | 65.42 | -87.84 | 146.40 |
| REINFORCE `gamma` | low | -785.90 | 387.48 | -1805.48 | -394.91 |
| REINFORCE `gamma` | opt | -737.28 | 392.15 | -1805.48 | -330.20 |
| REINFORCE `gamma` | high | **-217.84** | 157.12 | -533.69 | **32.32** |

**Interpretation:**
- DQN erzielt in allen sinnvollen Einstellungen deutlich bessere Returns als REINFORCE.
- Beim DQN ist `epsilon_decay=0.999` klar zu langsam (zu lange Exploration), während low/opt stabile positive Mittelwerte erreichen.
- Bei REINFORCE verbessert ein hohes `gamma` die Ergebnisse klar, bleibt aber im Mittel negativ.

### 3.2 Trainingsendstand (final moving average über 50 Episoden)

| Algorithmus | Konfiguration | final_moving_avg_50 | best_episode_reward |
|---|---|---:|---:|
| DQN | epsilon_decay_low | **217.41** | 310.86 |
| DQN | epsilon_decay_opt | 99.99 | **316.15** |
| DQN | epsilon_decay_high | -44.93 | 43.78 |
| DQN | learning_rate_low | 117.29 | 300.49 |
| DQN | learning_rate_opt | 99.99 | **316.15** |
| DQN | learning_rate_high | **159.52** | 295.26 |
| REINFORCE | gamma_low | -588.48 | 31.96 |
| REINFORCE | gamma_opt | -283.63 | 51.21 |
| REINFORCE | gamma_high | **-154.08** | **90.00** |
| REINFORCE | hidden_size_low | **-163.74** | 63.38 |
| REINFORCE | hidden_size_opt | -283.63 | 51.21 |
| REINFORCE | hidden_size_high | -387.32 | 15.87 |

**Interpretation:**
- DQN erreicht konsistent positive Lernstände (außer `epsilon_decay_high`).
- REINFORCE bleibt in allen betrachteten Läufen im negativen Bereich, zeigt aber mit `gamma_high` bzw. `hidden_size_low` vergleichsweise bessere Stabilisierung.

## 4) Plotbasierte Analyse

## 4.1 DQN – Einfluss von `epsilon_decay`
Verwendete Plots:
- `results/plots/dqn/dqn_epsilon_decay_low_seed1/reward.png`
- `results/plots/dqn/dqn_epsilon_decay_opt_seed1/reward.png`
- `results/plots/dqn/dqn_epsilon_decay_high_seed1/reward.png`
- zugehörige `moving_avg_reward.png`, `loss.png`, `epsilon.png`

Beobachtungen:
- **low (0.990):** schnelle Reduktion der Exploration, frühe Konsolidierung positiver Rewards.
- **opt (0.995):** balancierter Verlauf, einzelne starke Peaks, aber sichtbar höhere Varianz in späteren Episoden.
- **high (0.999):** Epsilon bleibt lange hoch, Lernfortschritt verzögert; Rewards bleiben häufig schwach/negativ.

Fazit: Für dieses Setup ist zu langsame Epsilon-Reduktion der kritischste DQN-Fehler.

## 4.2 DQN – Einfluss von `learning_rate`
Verwendete Plots:
- `results/plots/dqn/dqn_learning_rate_low_seed1/reward.png`
- `results/plots/dqn/dqn_learning_rate_opt_seed1/reward.png`
- `results/plots/dqn/dqn_learning_rate_high_seed1/reward.png`
- zugehörige `moving_avg_reward.png`, `loss.png`

Beobachtungen:
- **low (3e-4):** eher langsamer Lernstart, später stabil, aber nicht maximal performant.
- **opt (1e-3):** starke Einzelfolgen-Rewards, gleichzeitig deutliche Oszillation der Lernkurve.
- **high (3e-3):** überraschend solide Endperformance, aber je nach Phase sprunghafte Updates.

Fazit: Die Lernrate ist sensitiv, jedoch weniger destruktiv als ein ungeeignetes `epsilon_decay`.

## 4.3 REINFORCE – Einfluss von `gamma`
Verwendete Plots:
- `results/plots/reinforce/reinforce_gamma_low_seed1/reward.png`
- `results/plots/reinforce/reinforce_gamma_opt_seed1/reward.png`
- `results/plots/reinforce/reinforce_gamma_high_seed1/reward.png`
- zugehörige `moving_avg_reward.png`, `policy_loss.png`

Beobachtungen:
- **low (0.90):** sehr kurzsichtige Optimierung, häufige starke Abstürze, dauerhaft schlechter Return.
- **opt (0.99):** leichte Verbesserung gegenüber low, bleibt aber hochvariabel und negativ.
- **high (0.999):** beste REINFORCE-Variante im Vergleich; einzelne Episoden werden positiv abgeschlossen.

Fazit: Für LunarLander profitiert REINFORCE deutlich von langfristigerem Return-Horizont.

## 4.4 REINFORCE – Einfluss von `hidden_size`
Verwendete Plots:
- `results/plots/reinforce/reinforce_hidden_size_low_seed1/reward.png`
- `results/plots/reinforce/reinforce_hidden_size_opt_seed1/reward.png`
- `results/plots/reinforce/reinforce_hidden_size_high_seed1/reward.png`
- zugehörige `moving_avg_reward.png`, `policy_loss.png`

Beobachtungen:
- **low (64):** beste Stabilisierung im Vergleich der drei Hidden-Size-Läufe.
- **opt (128):** schwankender, insgesamt schwächer als low.
- **high (256):** keine Robustheitsgewinne sichtbar, tendenziell instabiler.

Fazit: Größere Policy-Netze helfen hier nicht automatisch; Optimierungsvarianz dominiert.

## 5) Videoanalyse
Es liegen für alle relevanten Konfigurationen Videos vor (u. a. unter `videos/dqn/*` und `videos/reinforce/*`, inkl. `eval10_seed100`).

Zusammenfassende qualitative Tendenzen (konsistent zu den Reward-Metriken):
- **Starke DQN-Konfigurationen** zeigen häufiger kontrollierte Sinkphasen und mehr erfolgreiche Landungen.
- **Schwache DQN-Einstellungen** (insb. `epsilon_decay_high`) zeigen häufiger chaotische Korrekturen und Fehllandungen.
- **REINFORCE-Konfigurationen** zeigen deutlich mehr inkonsistente Steuersequenzen, harte Landungen bzw. frühzeitige Crashes.

## 6) Gesamtvergleich DQN vs. REINFORCE

1. **Sample-Effizienz:** DQN lernt in diesem Setup klar effizienter und erreicht deutlich höhere Endperformance.
2. **Stabilität:** DQN zeigt zwar Oszillationen, stabilisiert aber bei geeigneter Exploration in den positiven Bereich.
3. **Varianz:** REINFORCE weist hohe Varianz und überwiegend negative Returns auf.
4. **Hyperparameter-Sensitivität:**
   - DQN ist besonders sensitiv auf `epsilon_decay`.
   - REINFORCE ist besonders sensitiv auf `gamma`; `hidden_size` wirkt sekundär.

## 7) Fazit
Auf Basis der vorhandenen Branch-Artefakte (Plots, Tabellen, Videos) ist **DQN der klare Gewinner** für `LunarLander-v3` in diesem Projektsetup. Die besten Ergebnisse stammen aus DQN-Läufen mit schneller bzw. mittlerer Epsilon-Reduktion. REINFORCE zeigt Lernfortschritt nur punktuell und bleibt im Mittel deutlich unter DQN.

## 8) Empfohlene nächste Schritte
- Mehrere Trainingsseeds pro Konfiguration (statistische Absicherung).
- **REINFORCE mit mehr Episoden nachziehen** (konkret empfohlen):
  - aktuelle Läufe nutzen 800 Episoden; als nächstes 1600 und 2400 Episoden testen,
  - Fokus auf `gamma=0.999` und `hidden_size=64` (beste REINFORCE-Trends in den vorhandenen Artefakten),
  - pro Setting mind. 3 Seeds und identische Eval-Protokolle, um Varianz sauber zu trennen.
- Längere Trainingsläufe möglichst mit Baseline/Advantage-Varianten kombinieren.
- Vergleich mit PPO/A2C als stabilere policy-gradient Baselines.
- Einheitliche Evaluationsprotokolle für **alle 12** Konfigurationen in `results/tables/`.
