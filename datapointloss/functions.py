import numpy as np

import scipy.integrate as integrate

import matplotlib.pyplot as plt 
import matplotlib.colors as mcolors
from matplotlib.legend_handler import HandlerTuple

def default(c):
  """ 
  Kaal, mis korrutatakse keskmise hinnaga. Erinevad kaalud 
  annavad erinevaid kõveraid, mille alune pindala on mingi rangelt
  korralik kadu. Default kaal tähendab kaalu puudumist, iga kaal on 1,
  w(c) = 1.

  Argumendid:
    float c: Hindade suhe

  Tagastab:
    int weight: 1
  """ 
  return 1

def brier_curve(y_true, p_pred, w=default, x=None):
  """ 
  Funktsioon, mis joonistab Brier'i kõvera.

  Argumendid:
    numpy.ndarray y_true: Andmepunktide tegelikud väärtused.

    numpy.ndarray p_pred: Andmepunktidele ennustatud positiivse klassi 
      tõenäosus.

    function w: Funktsioon, mis annab iga hindade suhte juures kaalu. Ehk
      millega korrutatkse keskmine hind iga hinna suhte juures läbi.
      Erinevad kaalud annavad kõveraid, mille alune pindala on erinev
      rangelt korralik kadu. Siin on kaasas kaalud default(c), brier(c)
      ja cross_entropy(c).

    list x: y_teljel olevate hindade suhted. Tavaline väärtus on 0ist 
      1ni 1000 osaks jaotatud.

  Tagastab: 
    numpy.ndarray y_space: Kõikidel x-i, ehk c väärtustel oleva keskmise 
    hinna w(c) * L(c, 1-c) järjendina.
  """ 
  if x is None:
    x = np.linspace(0.000,1.000,1001).tolist()

  y_space = np.array([])
  n = len(p_pred)
  for c in x:
    sum = 0
    for i in range(0, n):
      pos_prob = p_pred[i]
      actual_class = y_true[i]
      if pos_prob > c and actual_class == 0:
        sum += c
      elif pos_prob <= c and actual_class == 1:
        sum += 1 - c
    l = w(c) * (sum / n)
    y_space = np.append(y_space, l)
  return y_space

def brier(c):
  """ 
  Kaal, mis korrutatakse keskmise hinnaga. Erinevad kaalud 
  annavad erinevaid kõveraid, mille alune pindala on mingi rangelt
  korralik kadu. Brier'i kõver saadakse kaaluga 2, w(c) = 2.

  Argumendid:
    float c: Hindade suhe

  Tagastab:
    int weight: 2
  """ 
  return 2

## w(c) = 1 / (c * (1-c))
def cross_entropy(c):
  """ 
  Kaal, mis korrutatakse keskmise hinnaga. Erinevad kaalud 
  annavad erinevaid kõveraid, mille alune pindala on mingi rangelt
  korralik kadu. Moonutatud ristentroopia kõver saadakse kaaluga
  w(c) = 1 / (c * (1-c)).

  Argumendid:
    float c: Hindade suhe

  Tagastab:
    int weight: 1 / (c * (1-c))
  """ 
  return 1 / (c * (1-c))

class Feature:
  """ 
  Tunnuse klass, mis on abiks tunnuste eraldamiseliga andmepunkti kao pindalana joonistamisel.

  Argumendid:
    dict values: 
      Numbriline tunnus:
        Sõnastik, mis koosneb ühest elemendist, kus võti on one hot enkodeeritud 
        tunnuse X_test veeru nimi, mida tahetakse curve_areas funktsioonis 
        kasutatada andmepunkti pindala joonistamisel. Sellele võtmele vastav väärtus on vahemikud,
        mida eraldi tahetakse kujutada. Näiteks argumendi {"age": [0, 1, 2, 3]} korral värvitakse 
        eri värvi X_test veeru "age" väärtuste vahemikud 0-1, 1-2 ja 2-3.
        Võtmele vastab

      Kategooriline tunnus:
        Sõnastik, mis koosneb kahest või enamast elemendist, kus võti on one hot enkodeeritud 
        andmepunktide tunnuste X_test veeru nimi, mida tahetakse curve_areas funktsioonis 
        kasutatada andmepunkti pindala joonistamisel. Sellele võtmele vastav väärtus on tunnuse väärtuse
        nimetus, mis joonise legendis kuvatakse. Näiteks argumendi {"sex_male": "Male", "sex_female": "Female"} 
        korral värvitakse eri värvi ühe onehot enkodeeritud tunnuse "sex" väärtused "male" ja "female". Legendis
        märgistatkse värvid nimetustega "Male" ja "Female".

  Muutujad: 
    bool numeric: Kas tegemist on numbrilise või kategoorilise tunnusega.
    list name: Numbrilise tunnuse puhul tunnuse X_test veeru nimi.
    list names: Kategoorilise tunnuse puhul legendis kuvatavad nimetused.
    list values: 
      Numbriline tunnus: List vahemikutest.
      Kategooriline tunnus: List X_test veeru nimedest.
  """ 
  def __init__(self, values):
      if len(values) == 1:
          self.numeric = True
      else:
          self.numeric = False
      if self.numeric:
          self.name = list(values.keys())[0]
          self.values = list(values.values())[0]
      else:
          self.names = list(values.values())
          self.values = list(values.keys())

def curve_areas(X_test, y_true, p_pred, x=None, w=default, targets=[0], feature=None):
  """ 
  Funktsioon, mis joonistab iga andmepunkti panustatud pindala kaos ja
  väljastab matplotlib ploti.

  Argumendid:
    pandas.DataFrame X_test: Andmepunkti tunnused, kus iga kategooriline
      väärtus on one hot enkodeeritud.

    numpy.ndarray y_true: Andmepunktide tegelik väärtus.

    numpy.ndarray p_pred: Andmepunktidele ennustatud positiivse klassi 
      tõenäosus.

    list x: y_teljel olevate hindade suhted. Tavaline väärtus on 0ist 
      1ni 1000 osaks jaotatud.

    function w: Funktsioon, mis annab iga hindade suhte juures kaalu. Ehk
      millega korrutatkse keskmine hind iga hinna suhte juures läbi.
      Erinevad kaalud annavad kõveraid, mille alune pindala on erinev
      rangelt korralik kadu. Siin on kaasas kaalud default(c), brier(c)
      ja cross_entropy(c).

    list targets: List, kus on andmepunktide väärtused, mille pindalad 
      joonistatakse. 0 - tegelikult negatiivsed, 1 - tegelikult positiivsed.

    datapointloss.functions.Feature feature: Tunnuse klass, et eraldada erinevate
      tunnustega admepunktid. Lähemalt loe klassi kirjeldusest.
  """ 
  if x is None:
    x = np.linspace(0.000,1.000,1001).tolist()

  copy = X_test.copy()
  copy['prob'] = p_pred

  probs = [
            copy[np.array(y_true) == 0].sort_values('prob'),
            copy[np.array(y_true) == 1].sort_values('prob', ascending=False)]

  colors = list(mcolors.TABLEAU_COLORS.keys()) + list(mcolors.BASE_COLORS.keys())

  n = len(p_pred)
  
  if feature != None:
      value_labels = [[] for i in feature.values]
    
  for target in targets:
      n_prob = len(probs[target])
      prob = probs[target]
      for i, row in prob.iterrows():
          y = np.array([])
          for c in x:
              l = 0
              if target == 0:
                  if row['prob'] > c:
                      l = w(c) * (c / n) * n_prob
              elif target == 1:
                  if row['prob'] <= c:
                      l = w(c) * ((1 - c) / n) * n_prob
              y = np.append(y, l)
          n_prob -= 1

          if feature == None:
              plt.fill_between(x, 0, y)
          elif not feature.numeric:
              for i in range(len(feature.values)):
                  value_of_feature = row[f'{feature.values[i]}']
                  if value_of_feature == 1:
                      plot = plt.fill_between(x, 0, y, color=colors[i])
                      value_labels[i].append(plot)
          elif feature.numeric:
              for i in range(len(feature.values) - 1):
                  if feature.values[i] <= row[feature.name] < feature.values[i+1]:
                      plot = plt.fill_between(x, 0, y, color=colors[i])
                      value_labels[i].append(plot)
    
  if feature != None:
      value_labels = [tuple(i) for i in value_labels]
  
      values = None
      if feature.numeric:
          values = [ f'{str(feature.values[i])}<=...<{str(feature.values[i+1])}' for i in range(len(feature.values) - 1)]
      else:
          values = feature.names
  
      plt.legend(value_labels, values, handler_map={tuple: HandlerTuple(ndivide=None)})

def loss_bins(y_true, p_pred, x=None, w=default, target=0, bins=100):
  """ 
  Funktsioon, mis joonistab iga andmepunkti panustatud pindala kaos tulpdiagrammina.

  Argumendid:
    numpy.ndarray y_true: Andmepunktide tegelik väärtus.

    numpy.ndarray p_pred: Andmepunktidele ennustatud positiivse klassi 
      tõenäosus.

    list x: y_teljel olevate hindade suhted. Tavaline väärtus on 0ist 
      1ni 1000 osaks jaotatud.

    function w: Funktsioon, mis annab iga hindade suhte juures kaalu. Ehk
      millega korrutatkse keskmine hind iga hinna suhte juures läbi.
      Erinevad kaalud annavad kõveraid, mille alune pindala on erinev
      rangelt korralik kadu. Siin on kaasas kaalud default(c), brier(c)
      ja cross_entropy(c).

    int target: Andmepunkti väärtused, mis tulpdiagrammis joonistatakse.
      0 - tegelikult negatiivsed, 1 - tegelikult positiivsed.
      
    int bins: Mitmeks hindade suhte piirkonda andmepunktid jaotatakse 
      tulpadesse.
  """ 

  if x is None:
    x = np.linspace(0.000,1.000,1001).tolist()
  
  probs = []
  if target == 0:
      probs = np.sort(p_pred[y_true == 0])
  elif target == 1:
      probs = np.sort(p_pred[y_true == 1])[::-1]

  n_label = len(probs)

  n = len(p_pred)

  f = lambda c : w(c) * (1 / n) * np.sum(
    (1 if pred > c else 0) * (1 - target) * c +
    (0 if pred > c else 1) * target * (1 - c))

  losses = []

  for i in range(len(probs)):
      pred = probs[i]
      y, err = integrate.quad(f, 0, 1)
      losses.append([y])

  probs_list = []
  for i in range(len(probs)):
      probs_list.append([probs[i]])

  plt.hist(probs_list, weights=losses, bins=bins, histtype='stepfilled', stacked=True)

  plt.xlim([0, 1])

