# Perbandingan Model Klasifikasi pada Data Kesehatan

Proyek ini menampilkan analisis perbandingan berbagai model klasifikasi pada data kesehatan untuk memprediksi kelas target (PE/Non PE). Proyek ini mencakup tahap pra-pemrosesan data, seleksi fitur, pelatihan model, dan evaluasi performa. Model yang dibandingkan adalah Naive Bayes, K-Nearest Neighbors (KNN), dan Decision Tree. Alur kerja ini memperlihatkan performa model sebelum dan sesudah seleksi fitur, memberikan wawasan tentang dampak pengurangan fitur terhadap akurasi model.

## Daftar Isi

- [Ikhtisar Proyek](#ikhtisar-proyek)
- [Impor Library](#impor-library)
- [Memuat Dataset](#memuat-dataset)
- [Pra-pemrosesan Data](#pra-pemrosesan-data)
- [Mengatasi Nilai Hilang dan Pembersihan Data](#mengatasi-nilai-hilang-dan-pembersihan-data)
- [Enkode Data Kategorikal](#enkode-data-kategorikal)
- [Seleksi Fitur Menggunakan Recursive Feature Elimination (RFE)](#seleksi-fitur-menggunakan-recursive-feature-elimination-rfe)
- [Pemisahan Data Latih-Uji dan Standardisasi](#pemisahan-data-latih-uji-dan-standardisasi)
- [Pelatihan dan Evaluasi Model](#pelatihan-dan-evaluasi-model)
- [Analisis Perbandingan](#analisis-perbandingan)
- [Visualisasi Hasil](#visualisasi-hasil)

## Ikhtisar Proyek

Proyek ini menunjukkan alur kerja pemodelan machine learning yang umum menggunakan dataset kesehatan. Langkah-langkah yang dilakukan meliputi:
1. **Pra-pemrosesan Data**: Pembersihan data, mengatasi nilai hilang, dan mengonversi data kategorikal menjadi numerik.
2. **Seleksi Fitur**: Memilih fitur relevan menggunakan metode Recursive Feature Elimination (RFE).
3. **Pelatihan dan Evaluasi Model**: Melatih model, mengevaluasi dengan cross-validation, dan membandingkan performa pada data asli vs. fitur terpilih.

## Impor Library

Mengimpor pustaka yang dibutuhkan, termasuk `pandas`, `numpy`, `seaborn`, `matplotlib` untuk manipulasi dan visualisasi data, serta `scikit-learn` (`sklearn`) untuk pelatihan dan evaluasi model.

```python
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import RFE
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, classification_report
```

