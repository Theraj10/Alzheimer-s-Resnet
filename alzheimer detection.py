from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
# Define paths
train_dir = "C:\\Users\\thera\\Downloads\\archive (1)\\AugmentedAlzheimerDataset"
val_dir = "C:\\Users\\thera\\Downloads\\archive (1)\\OriginalDataset"

# Data augmentation and preprocessing
train_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)  # 80-20 split for training/validation

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset='training'
)

val_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset='validation'
)


# Load ResNet50 model pre-trained on ImageNet
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Freeze the base model layers
for layer in base_model.layers:
    layer.trainable = False

# Add custom classification layers
x = base_model.output
x = GlobalAveragePooling2D()(x)  # Replace Flatten with GAP for efficiency
x = Dense(128, activation='relu')(x)
x = Dense(64, activation='relu')(x)
output = Dense(4, activation='softmax')(x)  # 4 classes: Mild, Moderate, Non, Very Mild

# Build the final model
model = Model(inputs=base_model.input, outputs=output)

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

# Print the model summary
model.summary()
# Train the model
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=4,  # Adjust epochs based on performance
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    validation_steps=val_generator.samples // val_generator.batch_size
)
# Load test data
test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(
    val_dir,  
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    shuffle=False
)

# Evaluate the model
test_loss, test_accuracy = model.evaluate(test_generator)
print(f"Test Loss: {test_loss}")
print(f"Test Accuracy: {test_accuracy}")
model.save('resnet_alzheimer_model.h5')

# Load the model (if needed later)
from tensorflow.keras.models import load_model
loaded_model = load_model('resnet_alzheimer_model.h5')