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

# Data baru
data_baru = pd.DataFrame({
    'jam_belajar': [jam_belajar],
    'kehadiran': [kehadiran]
})

# Prediksi
hasil = model.predict(data_baru)

# Output
print("Hasil Prediksi:", hasil[0])