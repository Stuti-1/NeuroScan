import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
import pickle
import os
from datetime import datetime

from preprocess import load_data
from model import build_model

print("🧠 Brain Tumor Detection - Training Started")
print("=" * 50)

# Check if data directory exists
if not os.path.exists('../data/Training'):
    print("❌ ERROR: 'data/Training' directory not found!")
    print("Please make sure you have extracted the Kaggle dataset correctly.")
    exit(1)

print("📂 Loading data from 'data/Training'...")
data = load_data('../data/Training')

if len(data) == 0:
    print("❌ ERROR: No data loaded! Check your dataset structure.")
    exit(1)

print(f"✅ Successfully loaded {len(data)} images")

# Prepare features and labels
print("🔄 Preparing features and labels...")
X = []
y = []

for features, label in data:
    X.append(features)
    y.append(label)

X = np.array(X) / 255.0
y = np.array(y)

print(f"📊 Dataset shape: {X.shape}")
print(f"📊 Labels shape: {y.shape}")

# Check class distribution
unique, counts = np.unique(y, return_counts=True)
class_names = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary']
print("\n📈 Class Distribution:")
for i, (cls, count) in enumerate(zip(class_names, counts)):
    print(f"  {cls}: {count} images")

# Split data
print("\n🔀 Splitting data into train/validation sets...")
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

print(f"🎯 Training samples: {len(X_train)}")
print(f"🎯 Validation samples: {len(X_val)}")

# Build and compile model
print("\n🏗️  Building model...")
model = build_model()

print("📋 Model Summary:")
model.summary()

# Create models directory if it doesn't exist
os.makedirs('models', exist_ok=True)

# Train model
print("\n🚀 Starting training...")
print("=" * 50)

history = model.fit(
    X_train, y_train, 
    epochs=10, 
    validation_data=(X_val, y_val),
    batch_size=32,
    verbose=1  # This will show progress bars
)

# Save model
model_path = '../models/tumor_detector.h5'
print(f"\n💾 Saving model to {model_path}...")
model.save(model_path)

# Display final results
final_train_acc = history.history['accuracy'][-1]
final_val_acc = history.history['val_accuracy'][-1]
final_train_loss = history.history['loss'][-1]
final_val_loss = history.history['val_loss'][-1]

print("\n" + "=" * 50)
print("🎉 Training Complete!")
print(f"📊 Final Training Accuracy: {final_train_acc:.4f}")
print(f"📊 Final Validation Accuracy: {final_val_acc:.4f}")
print(f"📊 Final Training Loss: {final_train_loss:.4f}")
print(f"📊 Final Validation Loss: {final_val_loss:.4f}")
print(f"💾 Model saved successfully at: {model_path}")
print(f"⏰ Training completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# Quick test to ensure model loads correctly
print("\n🧪 Testing model loading...")
try:
    from tensorflow.keras.models import load_model
    test_model = load_model(model_path)
    print("✅ Model loads successfully!")
except Exception as e:
    print(f"❌ Error loading model: {e}")

print("\n🎯 Next steps:")
print("1. Run 'streamlit run app.py' to test your model")
print("2. Upload MRI images to get predictions")
print("3. Check the accuracy - if low, consider more epochs or data augmentation")