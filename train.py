import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam

# 1. Dataset Folders Check

train_dir = r"C:\Users\ad\OneDrive\Desktop\face_detection_recognization\archive (1)\images\images\train"
val_dir = r"C:\Users\ad\OneDrive\Desktop\face_detection_recognization\archive (1)\images\images\validation"


# 2. Data Generators for Training
train_datagen = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir, target_size=(48, 48), batch_size=64, color_mode='grayscale', class_mode='categorical'
)

validation_generator = val_datagen.flow_from_directory(
    val_dir, target_size=(48, 48), batch_size=64, color_mode='grayscale', class_mode='categorical'
)

# 3. Model Architecture Construction
model = Sequential([
    Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48, 48, 1)),
    Conv2D(64, kernel_size=(3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),

    Conv2D(128, kernel_size=(3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(128, kernel_size=(3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),

    Flatten(),
    Dense(1024, activation='relu'),
    Dropout(0.5),
    Dense(7, activation='softmax')
])

model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.0001), metrics=['accuracy'])

# 4. Actual Model Training (Laptop ke CPU par chalega)
print("🚀 AI Model Training is starting now...")
model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // 64,
    epochs=10, 
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // 64
)

# 5. Save the newly created model file automatically
model.save('emotion_model.h5')
print("✅ Training complete! 'emotion_model.h5' has been successfully created in your folder.")




