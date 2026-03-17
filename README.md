# Signal_Control
Adaptive Traffic Signal Control using Deep Reinforced Learning and Digital Twins based on OpenStreetMap (OSM) data.

# SUMO-RL Traffic Control Configuration Guide

Ten projekt implementuje inteligentny system sterowania ruchem (PPO Reinforcement Learning) w środowisku SUMO. Poniższa instrukcja pozwoli Ci skonfigurować środowisko od zera.

## 1. Wymagania wstępne (Prerequisites)

### System operacyjny: Windows 10/11
### Oprogramowanie:
1. **SUMO (Simulation of Urban MObility):** [Pobierz wersję 1.20.0+](https://sumo.dlr.de/docs/Downloads.php).
   - Zalecana instalacja w domyślnej ścieżce: `C:\Program Files (x86)\Eclipse\Sumo`.
2. **Python 3.8 - 3.14:** [Pobierz stąd](https://www.python.org/). 
   - *Ważne:* Podczas instalacji zaznacz opcję **"Add Python to PATH"**.

---

## 2. Konfiguracja zmiennych środowiskowych

Bez tego skrypty Pythona nie "dogadają się" z silnikiem symulacji.

1. Otwórz menu Start, wpisz **"Zmienne środowiskowe"** i wybierz "Edytuj zmienne środowiskowe systemu".
2. Kliknij **Zmienne środowiskowe**.
3. W sekcji **Zmienne systemowe** kliknij **Nowa**:
   - Nazwa: `SUMO_HOME`
   - Wartość: `C:\Program Files (x86)\Eclipse\Sumo` (lub Twoja ścieżka instalacji).
4. Znajdź zmienną `Path`, kliknij **Edytuj** -> **Nowy** i dodaj: `%SUMO_HOME%\bin`.

---

## 3. Instalacja projektu

Otwórz terminal (CMD lub PowerShell) w folderze projektu:

```bash
# 1. Tworzenie wirtualnego środowiska (zalecane)
python -m venv venv

# 2. Aktywacja środowiska
# Windows:
.\venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# 3. Instalacja bibliotek
pip install -r requirements.txt
