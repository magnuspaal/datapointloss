Minu bakalaureusetöö "Andmepunktide panuse visualiseeriminebinaarklassifitseerija" repositoorium. Kaustas notebooks/ on Jupyteri notebookid, kus on kõikide töös kasutatud visualisatsioonide joonistamisel kasutatud kood. Kaustas datapointloss/ on Pythoni teek, millega saab joonistatada Brier'i kõveraid, ristentroopia kõveraid ja tulpdiagramme, mida töös tutvustati. 

## Teegi instaleerimine
Teegi instaleerimiseks on vaja jooksutada peakaustas käsku:
```
pip install dist/datapointloss-1.0-py3-none-any.whl
```

## Teegi kompileerimine 
Teegi komplieerimiseks on vaja instaleerida teegid wheel, setuptools ja twine

```
pip install wheel
pip install setuptools
pip install twine
```
Seejärel saab kompileerimiseks jooksutada peakaustas käsku:
```
python setup.py bdist_wheel
```

## Importimine ja dokumentatsioon

Kui teek on instaleeritud, siis kõik teegi funktsioonid ja klassid saab Pythoni koodi importida käsuga:
```
from datapointloss.functions import brier_curve, curve_areas, loss_bins, brier, cross_entropy, default, Features
```
Samuti saab ka pydoc'i kasutades uurida iga funktsiooni ja klassi dokumentatsiooni. Näiteks funktsiooni brier_curve dokumentatsiooni saab uurida käsuga:
```
pydoc datapointloss.functions.brier_curve
```
