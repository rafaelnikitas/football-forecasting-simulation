# Football Forecasting & Simulation Lab

Probabilistic score forecasting, chronological backtesting and Monte Carlo season simulation.

[Read the case study](https://rafaelnikitas.github.io/studies/football-forecasting/) · [Full project documentation](football-forecasting-simulation/README.md) · [Notebook](football-forecasting-simulation/notebooks/01_end_to_end_forecasting_lab.ipynb) · [Model card](football-forecasting-simulation/MODEL_CARD.md) · [Portfolio](https://rafaelnikitas.github.io)

**Current evidence:** the engineering MVP is verified with deterministic synthetic data. Real EPL and La Liga benchmark results remain pending. Synthetic benchmark scores are not claims of real-football prediction performance.

## Start here

The implementation is in the `football-forecasting-simulation/` directory:

```bash
git clone https://github.com/rafaelnikitas/football-forecasting-simulation.git
cd football-forecasting-simulation/football-forecasting-simulation
python -m venv .venv
source .venv/bin/activate
pip install -e .
football-lab demo --output-dir reports/generated --simulations 2000
python -m unittest discover -s tests -v
```

For the optional interface, install `.[dashboard]` and run `streamlit run streamlit_app.py` from that same directory.

## What is included

- League-rate, independent Poisson and Dixon–Coles models.
- Expanding-window backtests with a temporal-leakage assertion.
- Log loss, Brier score, ranked probability score and calibration outputs.
- League-position simulation, provider adapters and a Streamlit demonstration.
- Unit tests, Docker, a model card and source-attribution guidance.

The [full README](football-forecasting-simulation/README.md) documents real-data evaluation commands and the boundary between checked software behaviour and evidence still to be generated.

Project code retains its existing MIT license. Provider data retains its own terms.
