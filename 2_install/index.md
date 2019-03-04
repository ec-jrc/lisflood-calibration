# How to install

## Install dependencies

To install Python 2.7 and PCRaster please refeer to official docs, according your platform (Linux, Windows, MacOS...)
[PCRaster 4.x](http://pcraster.geo.uu.nl/downloads/latest-release/)
[Python 2.7](https://www.python.org/downloads/)[^1]

## Clone the repository and install python requirements

```bash
git clone https://github.com/ec-jrc/lisflood-calibration.git
cd lisflood-calibration
pip2.7 install -r requirements.txt
```

After you cloned the repo and installed python dependencies you need to [setup](3_data) static data and configure the tool.

[^1]: This software will be soon ported to be compatible with Python > 3.5, as Python 2.7 won't be supported after 2019.
