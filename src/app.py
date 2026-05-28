import pandas as pd
from sklearn.neighbors import KNeighborsClassifier

# Membaca dataset
data = pd.read_csv('dataset/data_siswa.csv')

# Fitur
X = data[['jam_belajar', 'kehadiran']]

# Target
y = data['hasil']

# Membuat model KNN
model = KNeighborsClassifier(n_neighbors=3)

# Training model
model.fit(X, y)

print("=== SISTEM PREDIKSI NILAI SISWA ===")

# Input user
jam_belajar = float(input("Masukkan jam belajar: "))
kehadiran = float(input("Masukkan kehadiran: "))

# Prediksi
hasil = model.predict([[jam_belajar, kehadiran]])

# Output
print("Hasil Prediksi:", hasil[0])