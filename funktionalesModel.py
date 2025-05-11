import tensorflow as tf
import numpy as np
import random

# Funktion für das laden der csv Datei
def lade_csv_daten(pfad):
    with open(pfad, 'r') as datei:
        zeilen = datei.readlines()
    # mit 1: den Spaltennamen überspringen
    return zeilen[1:]



# Funktion für das Aufbereiten der Daten
# es braucht 5 leere Arrays, für jede Ziffer der 5-Stelligen Postleitzahl eins

def daten_verarbeiten(zeilen):
    # 5 Inputs, für jede Zahl einen
    x1, x2, x3, x4, x5 = [], [], [], [], []
    y1, y2, y3, y4, y5 = [], [], [], [], []

    random.shuffle(zeilen)

    for zeile in zeilen:
        # strip() um Leerzeichen am Anfang und Ende zu entfernen,
        # split(',') teilt in Liste mit Komma getrennt auf
        werte = zeile.strip().split(',')

        # label als Ausgangspunkt für 5 einzelnen Ziffern
        label = werte[0]

        # pixel ergibt sich auf dem NumPy-Array ab der Stelle 1 fortfolgend
        # dtype soll Vorzeichenlose IntegerWerte verwenden
        # 140 da 5 Ziffern nebeneinander (5*28=140)
        pixel = np.array(werte[1:], dtype=np.uint8).reshape(28, 140)

        # Einzelbilder (Ziffern) herausschneiden – je 28 Pixel breit
        x1.append(pixel[:, 0:28])
        x2.append(pixel[:, 28:56])
        x3.append(pixel[:, 56:84])
        x4.append(pixel[:, 84:112])
        x5.append(pixel[:, 112:140])

        # einzelne Ziffern als Integer zum Label hinzufügen
        y1.append((int(label[0])))
        y2.append((int(label[1])))
        y3.append((int(label[2])))
        y4.append((int(label[3])))
        y5.append((int(label[4])))



    """
    Umbau des Arrays ?????????????
    
    Funktion für Graustufenbilder → 4D-Tensor: (anzahl_bilder, höhe, breite, kanäle)
    -1 -> automatische Erkennung der Bildanzahl , 28x28 = Bildabmessungen, Kanäle = 1, weil Graustufenbild (kein RGB)
    
    """
    def reshape(x):
        return np.array(x).reshape(-1, 28, 28, 1)

    return [reshape(x1), reshape(x2), reshape(x3), reshape(x4), reshape(x5)], [np.array(y1), np.array(y2), np.array(y3), np.array(y4), np.array(y5)]
    print(x1.shape, x2.shape, x3.shape)  # zur Kontrolle der Dimensionsform

"""
Quelldaten laden

x_train, x_test: Bilder der Ziffern.
y_train, y_test: Entsprechende Labels (die Ziffern von 0 bis 9)

"""

# die Daten per Funktion von den .csv Dateien beschaffen:
train_zeilen = lade_csv_daten("mnist_train.csv")
x_train, [y1, y2, y3, y4, y5] = daten_verarbeiten(train_zeilen)

test_zeilen = lade_csv_daten("mnist_test.csv")
x_test, [yt1, yt2, yt3, yt4, yt5] = daten_verarbeiten(test_zeilen)


"""
Array-Dimensionen ermitteln und abspeichern
Datenstruktur analysieren
"""



"""
Das funktionales Modell erstellen, da das sequential Modell nur einen Output zulässt.
Benötigt werden jedoch 5 Outputs für die Postleitzahl.

als "Variable" für die Hidden-Schicht wird x verwendet. 
Da es sich bei Tensorflow um einen Datenfluss handelt wird x immer wieder abgeändert!
"""

# Starten mit dem Input
input = tf.keras.Input(shape=(28, 28, 1))

# 1. Schicht: Convolutional Layer
x = tf.keras.layers.Conv2D(16, kernel_size=(3, 3), activation='relu')(input)
x = tf.keras.layers.Conv2D(16, kernel_size=(3, 3), activation='relu')(x)

# 2. Schicht: Pooling Layer
x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x)

# zum "Flatten"
x = tf.keras.layers.Flatten()(x)

# abschließende Schichten: 5 Stück als Ausgabe der Postleitzahl mit Zahlen von 0 bis 9 -> 10 Units
output1 = tf.keras.layers.Dense(10, activation='softmax')(x)
output2 = tf.keras.layers.Dense(10, activation='softmax')(x)
output3 = tf.keras.layers.Dense(10, activation='softmax')(x)
output4 = tf.keras.layers.Dense(10, activation='softmax')(x)
output5 = tf.keras.layers.Dense(10, activation='softmax')(x)

# das Modell bilden:
modell_plz = tf.keras.Model(inputs=input, outputs=[output1,output2,output3,output4,output5])

"""
# das Modell für das Training vorbereiten.
Da nun 5 Outputs vorhanden sind muss auch 5 mal der Verlust/loss definiert werden. Einmal für jede Ziffer.
"""
modell_plz.compile(optimizer='adam', loss=['sparse_categorical_crossentropy'] * 5, metrics=['accuracy'])

# und trainieren
modell_plz.fit(x=x_train,y=[y1, y2, y3, y4, y5], epochs=1)

# evaluieren mit den Testlabels
modell_plz.evaluate(x_test, [yt1, yt2, yt3, yt4, yt5])

# das Modell speichern
modell_plz.save('postleitzahlen')


tf.keras.models.load_model("postleitzahlen")
