import tensorflow as tf
import numpy as np
import random

# Funktion für das Laden der csv Datei
def lade_csv_daten(pfad):
    # erste Zeile überspringen (Header), np.loadtxt -> lädt Datei als NumPy-Array
    daten = np.loadtxt(pfad, delimiter=",", skiprows=1)

    # erste Spalte = Labels (die Ziffer von 0–9)
    # uint8 konvertiert die Labels in vorzeichenlose ganzzahlige Werte
    labels = daten[:, 0].astype(np.uint8)

    # alle Zeilen/Bilder ; 1: -> alle Pixelwerte ab Index 1
    # -> es werde also alle Bilddaten gespeichert
    pixel = daten[:, 1:]

    # Aufruf der Funktion zur Datenaufbereitung, angewendet auf die "pixel"-Daten
    bilder = reshape_graustufenbilder(pixel)

    # gibt die aufbereiteten Bilddaten und den dazugehörigen Labels zurück
    return bilder, labels


# Funktion zur Datenaufbereitung:
# Ausgangslage: Bilddaten liegen nur als flacher Vektor (Zeile mit 784 Werten) vor

def reshape_graustufenbilder(bilddaten):

    # Normalisieren:
    # -1 -> automatische Erkennung der Bildanzahl
    # flache Bilddaten bzw. flacher Vektor (28x28 = 784 Werte) in 28x28 Graustufenbilder (Matrizen) umformen
    # Kanäle = 1 -> ein Farbkanal, weil Graustufenbild (kein RGB)
    # / 255 -> normalisiert die Werte (von 0–255 auf 0–1).
    return bilddaten.reshape(-1, 28, 28, 1).astype(np.float32) / 255.0


"""
Quelldaten laden

x_train, x_test: Bilder der Ziffern.
y_train, y_test: Entsprechende Labels (die Ziffern von 0 bis 9)

"""
x_train, y_train = lade_csv_daten("mnist_train.csv")
x_test, y_test = lade_csv_daten("mnist_test.csv")

print("Trainingsdaten:", x_train.shape, y_train.shape)
print("Testdaten:", x_test.shape, y_test.shape)

"""
Das sequential Modell erstellen:
"""
modell_plz = tf.keras.Sequential()

# 1. Schicht: Convolutional Layer, inkl. Input ; "Abtastung" mittels Kernel -> Anzahl 32 Filter mit Größe 3x3 Pixel
modell_plz.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))

# 2. Schicht: Pooling Layer, 2x2 Bereich im Bild wird angeschaut
modell_plz.add(tf.keras.layers.MaxPooling2D((2, 2)))

# 2. Hidden-Schicht hinzufügen, anschließend nochmals Pooling
modell_plz.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))
modell_plz.add(tf.keras.layers.MaxPooling2D((2, 2)))

# "Flatten" / abflachen wieder auf 1d Vektor
modell_plz.add(tf.keras.layers.Flatten())

# zusätzliche Dense-Schicht für bessere Genauigkeit
modell_plz.add(tf.keras.layers.Dense(64, activation='relu'))

# Finale Output-Schicht, 10 Units für Ziffern 0–9
modell_plz.add(tf.keras.layers.Dense(10, activation='softmax'))

# das Modell für das Training vorbereiten / kompilieren
modell_plz.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Zusammenfassung anzeigen
modell_plz.summary()

# und trainieren, gewählt sind 10 epochs/Durchläufe für bessere Genauigkeit nahe 1.0000
modell_plz.fit(x=x_train,y=y_train, epochs=10)

"""
Die Ausgabe in der Konsole
"""
# 5 zufällige Ziffern aus dem Testset, ergibt eine Postleitzahl
anzahl = 5
indizes = random.sample(range(len(x_test)), anzahl)

# Nur 5 Bilder herausziehen
x_test_auswahl = x_test[indizes]
y_test_auswahl = y_test[indizes]


# Vorhersage der 5 zufälligen Ziffern aus x_test
vorhersage = modell_plz.predict(x_test_auswahl)

# Die erkannten Ziffern als "Zahlenreihe" ausgeben
ziffernreihe = ''.join(str(np.argmax(i) for i in vorhersage))

"""
# Konsolenausgabe der erkannten Postleitzahl, argmax für den höchsten Wert:

Es werden mehrere Bilder aus der Vorhersage genommen und berechnet für jedes die vorhergesagte Ziffer
-> dann Umwandlung in String und verketten zu einer einzigen Ziffernreihe.
"""
print("Erkannte Ziffernreihe/Postleitzahl:", ''.join(str(np.argmax(i)) for i in vorhersage))
print("Tatsächlich:", ''.join(str(z) for z in y_test_auswahl))

# das Modell speichern
modell_plz.save('postleitzahlen')

# Laden des Modells
tf.keras.models.load_model("postleitzahlen")

# Evaluieren mit den Testlabels
modell_plz.evaluate(x_test,y_test)
