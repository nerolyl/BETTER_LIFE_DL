import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# إعداد معلمات أساسية
img_width, img_height = 150, 150
batch_size = 32
input_shape = (img_width, img_height, 3)

# 1. إنشاء نموذج CNN
model = models.Sequential()

# الطبقة الأولى - تلافيف مع 32 مرشح (3x3)
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
model.add(layers.MaxPooling2D((2, 2)))

# الطبقة الثانية - تلافيف مع 64 مرشح (3x3)
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))

# الطبقة الثالثة - تلافيف مع 128 مرشح (3x3)
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Dropout(0.5))  # إضافة طبقة Dropout بنسبة 50%


# Flatten لتحويل المصفوفات إلى متجه
model.add(layers.Flatten())

# الطبقة كاملة الاتصال
model.add(layers.Dense(512, activation='relu'))

# الطبقة النهائية - التصنيف
model.add(layers.Dense(7, activation='softmax'))  # 101 فئة للطعام

# 2. تجميع النموذج
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# ملخص النموذج
model.summary()
train_datagen = ImageDataGenerator(
    rescale=1.0/255,            # إعادة قياس القيم بين 0 و1
    rotation_range=40,          # تدوير الصور
    width_shift_range=0.2,      # تغيير العرض
    height_shift_range=0.2,     # تغيير الارتفاع
    shear_range=0.2,            # قص الصور
    zoom_range=0.2,             # تكبير/تصغير
    horizontal_flip=True,       # انعكاس أفقي
    fill_mode='nearest')        # ملء المساحات

test_datagen = ImageDataGenerator(rescale=1.0/255)

train_generator = train_datagen.flow_from_directory(
    'data/train',               # مسار البيانات التدريبية
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical')    # تصنيف متعدد الفئات

validation_generator = test_datagen.flow_from_directory(
    'data/valid',          # مسار بيانات التحقق
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical')
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size,
    epochs=25,                      # عدد الدورات
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // batch_size)
test_generator = test_datagen.flow_from_directory(
    'data/test',
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical')

test_loss, test_acc = model.evaluate(test_generator, steps=test_generator.samples // batch_size)
print(f'Test accuracy: {test_acc}')


import matplotlib.pyplot as plt

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'bo', label='Training accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend()

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()
