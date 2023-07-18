import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.utils import np_utils

# Veri setini yükleme ve ön işleme
# Örnek veri seti yerine kendi veri setinizi kullanmalısınız

# X: Müzik parçası özellikleri, y: Müzik türleri
X = np.load("features.npy")
y = np.load("labels.npy")

# Müzik türlerini sayısal etiketlere dönüştürme
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
num_classes = len(label_encoder.classes_)

# Veri setini eğitim ve test olarak bölme
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Sınıf etiketlerini kategorik hale dönüştürme
y_train_categorical = np_utils.to_categorical(y_train, num_classes)
y_test_categorical = np_utils.to_categorical(y_test, num_classes)

# Modeli oluşturma
model = Sequential()
model.add(Dense(256, input_shape=(X_train.shape[1],), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

# Modeli derleme
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Modeli eğitme
model.fit(X_train, y_train_categorical, validation_data=(X_test, y_test_categorical), epochs=50, batch_size=32)

# Tahmin yapma
predictions = model.predict(X_test)
predicted_labels = np.argmax(predictions, axis=1)
predicted_labels = label_encoder.inverse_transform(predicted_labels)

# Sonuçları değerlendirme
accuracy = np.sum(predicted_labels == y_test) / len(y_test)
print("Doğruluk: {:.2f}%".format(accuracy * 100))
