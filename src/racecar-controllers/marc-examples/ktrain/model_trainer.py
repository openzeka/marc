#!/usr/bin/env python
# coding: utf-8

# matris işlemleri için
import numpy as np
# opencv 
import cv2
# grafik kütüphanesi 
import matplotlib.pylab as plt
# rasgele sayı üretimi için
import random


# Eğitim için kaydettiğimiz training.py dosyasını okuyoruz
PATH='../deep_learning/data/006/'
data = np.load(PATH+'training_data.npy')

print('Toplam veri sayısı:', len(data))
print(data[0])
# Kayıt ettğimiz resmin birine bakıyoruz
image = cv2.imread(PATH+data[30][0])
plt.figure(figsize=(15,5))
plt.imshow(image)


image_resized = cv2.resize(image, (320,180))
plt.figure(figsize=(15,5))
plt.imshow(image_resized[100:,:,:]) #Cropped image 


images = list(img[0] for img in data[1:])
labels = list(float(img[2]) for img in data[1:])


# Verimizdeki açıların dağılımı nasıl diye bir histogram yapıp bakıyoruz
# Dağılımın eşit şekilde olmaması eğitimin de düzgün olmamasına sebep olur
plt.hist(labels)
plt.show()


# Veri setindeki açı dağılımını bir paröa düzeltmek için
# sayısı az olan açıdaki kayıtları listeye yeniden ekleyerek 
# daha düzgün hale getirmeye çalışıyoruz

nitem = len(images)
for i in range(nitem):
    if labels[i] > 0.05:
        for j in range(7):
            images.append(images[i])
            labels.append(labels[i])    
    if labels[i] < -0.07:
        for j in range(2):
            images.append(images[i])
            labels.append(labels[i]) 



# İlk histgorama göre daga dengeli sayılabilecek bir dağılıma ulaştık
# En doğru çözüm değil ama pratik işe yarar bir alternatif
plt.hist(labels)
plt.show()


# In[ ]:
print('Toplam resim sayısı: ', len(images))
print('Toplam etiket sayısı: ', len(labels))



# Veri setimiz ile ilgili ayarlamalar
# Veri seti küme büyüklüğü batch size
# Verisetinin ne kadarı eğitim ne kadarı test için kullanılacak
# Eğitim %80 , Test %20 
bsize = 8
dlen = len(labels)
splitpoint = int(0.8*dlen)
reindex = list(range(len(labels)))
# Eğtim verisini karıştıryoruz
random.seed(1234)
random.shuffle(reindex)


# In[ ]:


# Resim üzerinde Rastgele parlaklık değişimi uygulayan bir fonksiyon
# Augmentation function (taken from github)
def augment_brightness(image):
    image1 = cv2.cvtColor(image,cv2.COLOR_BGR2HSV) 
    image1 = np.array(image1, dtype = np.float64)
    random_bright = .5+np.random.uniform()
    image1[:,:,2] = image1[:,:,2]*random_bright
    image1[:,:,2][image1[:,:,2]>255]  = 255
    image1 = np.array(image1, dtype = np.uint8)
    image1 = cv2.cvtColor(image1,cv2.COLOR_HSV2RGB)
    return image1


# In[ ]:


# ismi verilen resmi okuyup 
# rastgele olarak %50 sine parlaklık değişimi uygulayan fonksiyonu uygulayıp
# resim matrisini dönem bir fonksiyon

def get_matrix(fname):
    img = cv2.imread(PATH+fname)
    img = cv2.resize(img, (320,180))
    if random.randint(0,1) == 1 :
        img = augment_brightness(img)        
    return img[100:,:,:] # Return the cropped image, (320,80)


# In[ ]:


# Bütün veriyi hafızaya almamız mümkün değil
# Ek olarak bazen çeşitli değişimler - Augmentation - uygulamakda istiyebiliriz
# python generator ile gerektiğinde veri okunur düzenlenir ve eğitim veya test için 
# sisteme verilir
# alttaki fonksiyonlar bu işi yapar

# Generate data for training
def generate_data():
    i = 0
    while True:
        x = []
        y = []
        for j in range(i,i+bsize):  
            ix = reindex[j]
            img = get_matrix(images[ix])
            lbl = np.array([labels[ix]])
            flip = random.randint(0,1)
            if flip == 1:
                img = cv2.flip(img,1)
                lbl = lbl*-1.0
            x.append(img)
            y.append(lbl)
        x = np.array(x)
        y = np.array(y)       
        yield (x,y)    
        i +=bsize
        if i+bsize > splitpoint:
            i = 0
            
# Generate data for validation                  
def generate_data_val():
    i = splitpoint
    while True:
        x = []
        y = []
        for j in range(i,i+bsize): 
            ix = reindex[j]
            x.append(get_matrix(images[ix]))
            y.append(np.array([labels[ix]]))
        x = np.array(x)
        y = np.array(y)       
        yield (x,y)    
        i +=bsize
        if i+bsize > dlen:
            i = splitpoint


# In[ ]:


# Keras için gerekenler
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Lambda
from keras.layers import Conv2D, MaxPooling2D, Cropping2D
from keras.callbacks import ModelCheckpoint
from keras.optimizers import SGD


# In[ ]:


# Model based on NVIDIA's End to End Learning for Self-Driving Cars model
# Sıralı bir keras modeli tanılıyoruz
model = Sequential()
# Normalization
# 0 - 255 arası değerler -1 ila 1 arasına çekiliyor
model.add(Lambda(lambda x: x / 127.5 - 1.0, input_shape=(80, 320, 3)))
# Evrişim katmanı (5, 5) lik 24 tane 2 şer piksel kayarak
model.add(Conv2D(24, (5, 5), activation="relu", strides=(2, 2)))
model.add(Conv2D(36, (5, 5), activation="relu", strides=(2, 2)))
model.add(Conv2D(48, (5, 5), activation="relu", strides=(2, 2)))
model.add(Conv2D(64, (3, 3), activation="relu"))
model.add(Conv2D(64, (3, 3), activation="relu"))
# Ağın çıkışı burda vectöre çevriliyor
model.add(Flatten())
# Yapay Sinir ağı kısmı
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
# Ağın çıkışı Açı 
model.add(Dense(1))
model.compile(loss='mse', optimizer='adam')


# In[ ]:


# Tanımladığımız ağın yapsı
model.summary()


# In[ ]:


# Açı değerlerinide -0.3 ila 0.3 aralığından -1 ila 1 aralığına çekebilmek için 3 ile çarpıyoruz
labels = 3*np.array(labels)


# In[ ]:


# Eğitim esnasında test hata değeri en düşük değeri kaydeden bir fonksiyon
model_checkpoint = ModelCheckpoint('weights.h5', monitor='val_loss', save_best_only=True)


# In[ ]:


# Eğitim fonksiyonu 
hs = model.fit_generator(generate_data(),steps_per_epoch=int(splitpoint/ bsize),
                    validation_data=generate_data_val(), 
                    validation_steps=(dlen-splitpoint)/bsize, epochs=10,callbacks=[model_checkpoint])


# In[ ]:


# Eğittiğimiz modeli kaydediyoruz
# Ağ yapsını json olarak
# Ağ parametre değerlerini h5 uzantılı olarak
import json 
# Save model weights and json.
mname = 'model_new'
model.save_weights(mname+'.h5')
model_json  = model.to_json()
with open(mname+'.json', 'w') as outfile:
    json.dump(model_json, outfile)


# In[ ]:


# rastgele 10 resim seçip modelimiz hesapladığı sonuçla gerçeğine bakıyoruz 
# Eğer sonuçlar iyi ise kullanabiliriz
# Sonuççlar kötüyse eğitim aşamasına dönmemiz lazım
# Compare actual and predicted steering
for i in range(10):
    ix = random.randint(0,len(df)-1)
    out = model.predict(get_matrix(images[ix]).reshape(1,80,320,3))
    print(labels[ix], ' - > ', out[0][0])


# In[ ]:




