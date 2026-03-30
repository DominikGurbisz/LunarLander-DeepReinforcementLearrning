# Report Template – LunarLander (DQN vs. REINFORCE)

## 1. Einleitung
- Ziel der Arbeit: Vergleich eines value-based (DQN) und eines policy-gradient (REINFORCE) Verfahrens auf LunarLander.
- Kurz: Warum ist LunarLander ein geeignetes Benchmark-Problem?

## 2. Umgebung und Problemstellung
- Gymnasium `LunarLander-v3`, diskreter Aktionsraum.
- Reward-Mechanik (grobe Beschreibung), Episodenende (`terminated`/`truncated`).

## 3. Methoden
### 3.1 DQN
- Q-Network, Target-Netzwerk, Replay Buffer, epsilon-greedy, Mini-Batches.
- Update-Regel (Bellman-Target) und verwendete Loss.

### 3.2 REINFORCE
- Policy-Netzwerk, Sampling aus Aktionsverteilung.
- Discounted Returns und Policy-Gradient-Update.
- Optional Return-Normalisierung zur Stabilisierung.

## 4. Implementierungsdetails
- PyTorch, Device-Wahl (CPU/CUDA), Seeding-Konzept.
- Logging (CSV/JSON), Modell-Checkpoints, Evaluationspipelines.
- Videoaufzeichnung mit `RecordVideo`.

## 5. Hyperparameter-Design

### DQN
1. **epsilon_decay** (algorithmusrelevant)
   - low / opt / high
2. **learning_rate** (frei)
   - low / opt / high

### REINFORCE
1. **gamma** (algorithmusrelevant)
   - low / opt / high
2. **hidden_size** (frei)
   - low / opt / high

Begründe jeweils kurz, warum diese Werte sinnvolle Kontraste liefern.

## 6. Experiment-Setup
- Anzahl Episoden pro Lauf, Seeds, Evaluationssetting (Anzahl Episoden).
- Nenne die 12 Kernkonfigurationen explizit.

## 7. Ergebnisse: DQN Hyperparameter 1
- Plot: reward + moving average für `epsilon_decay` low/opt/high.
- Beobachtung:
  - Exploration zu kurz/lang?
  - Instabile Kurven?
  - Typische Fehlverhalten im Video (Crashs, Schwingen, zielloses Schweben).

## 8. Ergebnisse: DQN Hyperparameter 2
- Plot: reward/loss für `learning_rate` low/opt/high.
- Beobachtung:
  - Zu hohe LR: starke Schwankung/divergente Updates?
  - Zu niedrige LR: langsames Lernen?

## 9. Ergebnisse: REINFORCE Hyperparameter 1
- Plot: reward + policy_loss für `gamma` low/opt/high.
- Beobachtung:
  - Kurzsichtige Policy bei low gamma?
  - Höhere Varianz/instabile Gradienten bei extremen Einstellungen?

## 10. Ergebnisse: REINFORCE Hyperparameter 2
- Plot: reward/policy_loss für `hidden_size` low/opt/high.
- Beobachtung:
  - Unterkapazität vs. Überkapazität/Trainingsstabilität.

## 11. Vergleich der Algorithmen
- Sample-Effizienz, Stabilität, Varianz, Endperformance.
- Wie unterscheiden sich die Lernkurven?
- Welche Hyperparameter waren am sensitivsten?

## 12. Videoanalyse (Pflichtbezug)
Für jede relevante Hyperparameter-Einstellung kurz dokumentieren:
- Weiche Landung vs. harte Landung/Crash.
- Oszillation, chaotische Steuerung, Treibstoffintensität.
- Verhalten nahe Boden (Kontrolle/Übersteuerung).
- Konsistenz über mehrere Episoden.

## 13. Fazit
- Wichtigste Erkenntnisse zu DQN vs. REINFORCE auf LunarLander.
- Welche Einstellungen sind robust?
- Limitationen und mögliche nächste Schritte (z. B. mehr Seeds, PPO/A2C Vergleich).

---

## Vorschläge für Tabellen
1. **Hyperparameter-Tabelle**: low/opt/high je Algorithmus.
2. **Evaluationstabelle**: mean/std reward pro Konfiguration.
3. **Video-Beobachtungstabelle**: qualitative Symptome pro Konfiguration.

## Vorschläge für Pflichtplots
- reward pro Episode
- moving average reward
- DQN: loss + epsilon
- REINFORCE: policy_loss + return-Statistiken
- Vergleich low/opt/high je Hyperparameter in konsistenter Darstellung
