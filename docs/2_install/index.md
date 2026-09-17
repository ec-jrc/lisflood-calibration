# How to install

## Install dependencies

To install Python and the LISFLOOD hydrological model, please refer to https://github.com/ec-jrc/lisflood-code.

## Clone the repository and install the liscal Python library and its dependencies

```bash
git clone https://github.com/ec-jrc/lisflood-calibration.git
cd lisflood-calibration
pip install .
```

## Optional: interactive explorers

The interactive map explorers used by the diagnostics scripts (CAL_8 and CAL_9) rely on the hydrological analysis toolkit (`hat`). This package is not published on PyPI, so it is not installed automatically with `liscal` and must be installed manually:

```bash
pip install "hydro_analysis_toolkit @ git+https://github.com/ecmwf/hat.git@f94c313731953b011682ab2c49374c313f99791e"
```

After you cloned the repo and installed python dependencies you need to [setup](../3_data/index.md) static data and configure the tool.

