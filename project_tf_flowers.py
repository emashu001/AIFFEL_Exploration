#!/usr/bin/env python
# coding: utf-8

# # Step 1. 데이터셋 내려받기

# In[1]:


import warnings
warnings.filterwarnings("ignore")

print("완료!")


# In[2]:


import tensorflow as tf
print(tf.__version__)


# In[3]:


import tensorflow_datasets as tfds

tfds.__version__


# In[4]:


get_ipython().system('mkdir -p ~/aiffel/Exploration_CR4/tf_flowers')


# In[5]:


(raw_train, raw_validation, raw_test), metadata = tfds.load(
    name='tf_flowers',
    split=['train[:80%]', 'train[80%:90%]', 'train[90%:]'],
    data_dir='~/aiffel/Exploration_CR4/tf_flowers',
    download=False,
    with_info=True,
    as_supervised=True,
)


# In[6]:


print(raw_train)
print(raw_validation)
print(raw_test)


# # Step 2. 데이터셋을 모델에 넣을 수 있는 형태로 준비하기

# In[7]:


# 데이터시각화
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'retina'")


# In[8]:


plt.figure(figsize=(10, 5))

get_label_name = metadata.features['label'].int2str

for idx, (image, label) in enumerate(raw_train.take(10)):  # 10개의 데이터를 가져 옵니다.
    plt.subplot(2, 5, idx+1)
    plt.imshow(image)
    plt.title(f'label {label}: {get_label_name(label)}')
    plt.axis('off')


# In[9]:


IMG_SIZE = 160 # 리사이징할 이미지의 크기

def format_example(image, label):
    image = tf.cast(image, tf.float32)  # image=float(image)같은 타입캐스팅의  텐서플로우 버전입니다.
    image = (image/127.5) - 1 # 픽셀값의 scale 수정
    image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))
    return image, label


# In[10]:


train = raw_train.map(format_example)
validation = raw_validation.map(format_example)
test = raw_test.map(format_example)

print(train)
print(validation)
print(test)


# In[11]:


plt.figure(figsize=(10, 5))

get_label_name = metadata.features['label'].int2str

for idx, (image, label) in enumerate(train.take(10)):  # 10개의 데이터를 가져 옵니다.
    plt.subplot(2, 5, idx+1)
    plt.imshow(image)
    plt.title(f'label {label}: {get_label_name(label)}')
    plt.axis('off')


# # Step 3. 모델 설계하기

# In[12]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D


# In[13]:


model = Sequential([
    Conv2D(filters=16, kernel_size=3, padding='same', activation='relu', input_shape=(160, 160, 3)),
    MaxPooling2D(),
    Conv2D(filters=32, kernel_size=3, padding='same', activation='relu'),
    MaxPooling2D(),
    Conv2D(filters=64, kernel_size=3, padding='same', activation='relu'),
    MaxPooling2D(),
    Flatten(),
    Dense(units=512, activation='relu'),
    Dense(units=5, activation='softmax')
])


# In[14]:


model.summary()


# In[15]:


learning_rate = 0.0001
model.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=learning_rate),
              loss=tf.keras.losses.sparse_categorical_crossentropy,
              metrics=['accuracy'])


# # Step 4. 모델 학습시키기

# In[16]:


BATCH_SIZE = 32
SHUFFLE_BUFFER_SIZE = 1000


# In[17]:


train_batches = train.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)
validation_batches = validation.batch(BATCH_SIZE)
test_batches = test.batch(BATCH_SIZE)


# In[18]:


for image_batch, label_batch in train_batches.take(1):
    break

image_batch.shape, label_batch.shape


# In[19]:


validation_steps = 20
loss0, accuracy0 = model.evaluate(validation_batches, steps=validation_steps)

print("initial loss: {:.2f}".format(loss0))
print("initial accuracy: {:.2f}".format(accuracy0))


# In[20]:


EPOCHS = 10
history = model.fit(train_batches,
                    epochs=EPOCHS,
                    validation_data=validation_batches)


# # Step 5. 모델 성능 평가하기

# In[21]:


acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss=history.history['loss']
val_loss=history.history['val_loss']

epochs_range = range(EPOCHS)

plt.figure(figsize=(12, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend()
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend()
plt.title('Training and Validation Loss')
plt.show()


# In[22]:


for image_batch, label_batch in test_batches.take(1):
    images = image_batch
    labels = label_batch
    predictions = model.predict(image_batch)
    break

predictions


# In[23]:


import numpy as np
predictions = np.argmax(predictions, axis=1)
predictions


# In[24]:


plt.figure(figsize=(20, 12))

for idx, (image, label, prediction) in enumerate(zip(images, labels, predictions)):
    plt.subplot(4, 8, idx+1)
    image = (image + 1) / 2
    plt.imshow(image)
    correct = label == prediction
    title = f'real: {label} / pred :{prediction}\n {correct}!'
    if not correct:
        plt.title(title, fontdict={'color': 'red'})
    else:
        plt.title(title, fontdict={'color': 'blue'})
    plt.axis('off')


# In[25]:


count = 0   # 정답을 맞춘 개수
for image, label, prediction in zip(images, labels, predictions):
    # [[YOUR CODE]]
    correct = label == prediction
    if correct:
        count = count + 1

print(count / 32 * 100)


# # Step 4. 모델 학습시키기 - Transter Learning

# In[26]:


IMG_SHAPE = (IMG_SIZE, IMG_SIZE, 3)

# Create the base model from the pre-trained model VGG16
base_model = tf.keras.applications.VGG16(input_shape=IMG_SHAPE,
                                         include_top=False,
                                         weights='imagenet')


# In[27]:


image_batch.shape


# In[28]:


feature_batch = base_model(image_batch)
feature_batch.shape


# In[29]:


base_model.summary()


# In[30]:


feature_batch.shape


# In[31]:


# Global Average Pooling 계층을 만들기.
global_average_layer = tf.keras.layers.GlobalAveragePooling2D()


# In[32]:


feature_batch_average = global_average_layer(feature_batch)
print(feature_batch_average.shape)


# In[33]:


dense_layer1 = Dense(units=512, activation='relu')
dense_layer = Dense(units=52, activation='relu')
prediction_layer = Dense(units=5, activation='softmax')

# feature_batch_averag가 dense_layer를 거친 결과가 다시 prediction_layer를 거치게 되면
prediction_batch = prediction_layer(dense_layer(feature_batch_average))  
print(prediction_batch.shape)


# In[34]:


base_model.trainable = False


# In[35]:


model_vgg16 = tf.keras.Sequential([
  base_model,
  global_average_layer,
  dense_layer1,
  dense_layer,  
  prediction_layer
])


# In[36]:


model_vgg16.summary()


# In[37]:


base_learning_rate = 0.0001
model_vgg16.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=base_learning_rate),
              loss=tf.keras.losses.sparse_categorical_crossentropy,
              metrics=['accuracy'])


# In[38]:


validation_steps=20
loss0, accuracy0 = model_vgg16.evaluate(validation_batches, steps = validation_steps)

print("initial loss: {:.2f}".format(loss0))
print("initial accuracy: {:.2f}".format(accuracy0))


# In[39]:


EPOCHS = 10   # 이번에는 이전보다 훨씬 빠르게 수렴되므로 5Epoch이면 충분합니다.
history_vgg16 = model_vgg16.fit(train_batches,
                    epochs=EPOCHS,
                    validation_data=validation_batches)


# # Step 5. 모델 성능 평가하기 - Transfer Learning

# In[40]:


acc = history_vgg16.history['accuracy']
val_acc = history_vgg16.history['val_accuracy']

loss = history_vgg16.history['loss']
val_loss = history_vgg16.history['val_loss']

plt.figure(figsize=(16, 8))
plt.subplot(1, 2, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.ylabel('Accuracy')
plt.ylim([min(plt.ylim()),1])
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.ylabel('Cross Entropy')
plt.ylim([0,1.0])
plt.title('Training and Validation Loss')
plt.xlabel('epoch')
plt.show()


# In[41]:


for image_batch, label_batch in test_batches.take(1):
    images = image_batch
    labels = label_batch
    predictions = model.predict(image_batch)
    pass

predictions


# In[42]:


import numpy as np
predictions = np.argmax(predictions, axis=1)
predictions


# In[43]:


plt.figure(figsize=(20, 12))

for idx, (image, label, prediction) in enumerate(zip(images, labels, predictions)):
    plt.subplot(4, 8, idx+1)
    image = (image + 1) / 2
    plt.imshow(image)
    correct = label == prediction
    title = f'real: {label} / pred :{prediction}\n {correct}!'
    if not correct:
        plt.title(title, fontdict={'color': 'red'})
    else:
        plt.title(title, fontdict={'color': 'blue'})
    plt.axis('off')


# In[44]:


count = 0
for image, label, prediction in zip(images, labels, predictions):
    correct = label == prediction
    if correct:
        count = count + 1

print(count / 32 * 100) # 약 95% 내외


# # Step 6. 모델 활용하기

# In[45]:


import os
img_dir_path = os.getenv("HOME") + "/aiffel/Exploration_CR4/tf_flowers/images"
os.path.exists(img_dir_path)


# In[46]:


from tensorflow.keras.preprocessing.image import load_img, img_to_array


# In[47]:


# image를 load & arrary로 변환(reshape) & predict & percentage

def show_and_predict_image(dirpath, filename, img_size=160):
    filepath = os.path.join(dirpath, filename)
    image = load_img(filepath, target_size=(img_size, img_size))
    plt.imshow(image)
    plt.axis('off')
    image = img_to_array(image).reshape(1, img_size, img_size, 3)
    predictions = (model_vgg16.predict(image)[0])
    prediction_idx = np.argmax((model_vgg16.predict(image)[0]))
    
    flower_lists = metadata.features['label'].names
   
    print(f"This image seems {flower_lists[prediction_idx]} with {np.max(predictions) * 100}%.")


# In[48]:


filename = 'daisy.jfif'
show_and_predict_image(img_dir_path, filename)


# In[49]:


filename = 'dandelion.jpg'
show_and_predict_image(img_dir_path, filename)


# In[50]:


filename = 'roses.png'
show_and_predict_image(img_dir_path, filename)


# In[51]:


filename = 'sunflowers.jpg'
show_and_predict_image(img_dir_path, filename)


# In[ ]:




