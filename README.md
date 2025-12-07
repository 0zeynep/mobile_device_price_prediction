# Mobile Device Price Prediction
## Projenin Amacı

Bu projenin temel amacı, mobil cihaz fiyatlarını (Launched Price (USA)) etkileyen özellikleri sırayla deneyerek en uygun özelliği seçip fiyatla arasındaki ilişkiyi incelemek ve en uygun regresyon modelini uygulayarak fiyat tahmini yapmaktır.

## Kullanılan Veri Seti

Telefonlara ait:

Marka

Model

RAM

Ağırlık

Kamera çözünürlükleri

Ekran boyutu

Batarya kapasitesi

Farklı ülkelere göre fiyatlar

Çıkış yılı

gibi özellikleri içermektedir.


Projenin adımları şu şekildedir:


### 1)Veri setinin yüklenmesi:

        import pandas as pd
        import matplotlib.pyplot as plt
        import numpy as np

        mobiles=pd.read_csv("mobiles_dataset.csv",encoding='latin-1')
        mobiles.head()

Veri temizliği için pandas,regresyon sonuçlarını ve korelasyon matrisini gösterebilmek için  matplotlib,gerekli hesaplamalar için numpy kütüphanesi kullanılmıştır.


### 2)Fiyat Temizleme


    x = re.sub(r"[A-Za-z\s]+", "", x)
    x = x.replace(",", "")
    
USD,PKR gibi para birimi sembollerini,harfleri,virgülleri kaldırır.Sayısal olmayan bütün ifadeler kaldırılır.

    price_columns = [col for col in mobiles.columns if "Price" in col]

Price içeren bütün sütunları tespit eder ve bir liste oluşturur.

    for col in price_columns: mobiles[col] = mobiles[col].apply(clean_price_fixed_FINAL)

Bir döngü (loop) ile tespit edilen her bir fiyat sütununa sırayla girilir.

apply(clean_price_fixed_FINAL) komutu, o sütundaki her değere temizleme fonksiyonunu uygulayarak orijinal metin verilerini temizlenmiş sayısal (float) verilerle değiştirir.

    threshold=5000
    mobiles = mobiles[mobiles['Launched Price (USA)'] < threshold].copy()


Ayrık değerler tespit edilir.5000 in üstündeki değerler çıkarılır.

### 3)Bağımsız Özellikleri Temizleme

RAM,Mobile Weight,Battery Capacity,Front Camera,Back Camera,Screen Size özelliklerindeki sayısal olmayan ifadeler kaldırılmıştır.

### 4)Hedef Belirleme 

Bağımlı değişken (y) ,daha evrensel bir para birimi olduğu için  Launched Year(USA) olarak belirlenmiştir.

Bağımsız değişken(x) ise diğer özelliklerin hepsi karşılaştırılarak en uygun özellik RAM olarak belirlenmiştir.

### 5) Korelasyon Matrisi
   
Korelasyon matrisi,iki veya daha fazla değişkenin ne kadar güçlü ve ne yönde ilişkili olduğunu gösteren istatistiksel bir araçtır.’ değişkenin arasındaki katsayı 1,e ne kadar yakınsa bu onların arasında güçlü ilişki olduğunu gösterir. 

Isı haritasında da görüldüğü gibi en uygun ilişki Launched Price(USA) ile  RAM dir.

![heatmap](images/heatmapp.png)

### 6)RAM miktarı ile telefonların ortalama lansman fiyatı arasındaki ilişkiyi gösteren bar grafiği

Aşağıdaki bar grafiğinde de görüldüğü gibi RAM miktarı arttıkça Launched Price(USA) de artmaktadır.
![Bar Grafiği](images/bar.png)

### 7)Regresyon Uygulama

### Linear Regresyon

#### Hazırlık ve Veri Bölme

    ram_price_df = mobiles[[feature, target]].copy()

Sadece RAM ve Launched Price (USA) sütunlarını içerir.

    clean_subset = ram_price_df.dropna()

Seçilen bu iki sütundan herhangi birinde eksik (NaN) değere sahip olan tüm satırları veri setinden çıkarır.

    X = clean_subset[feature].to_numpy().reshape(-1, 1)

RAM sütununu alır ve scikit-learn kütüphanesinin ihtiyacı olan 2 boyutlu matris (reshape(-1, 1)) formatına dönüştürür.

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42
        )

Veri setini rastgele olarak %80 Eğitim (_train) ve %20 Test (_test) kümelerine ayırır. random_state=42 ise bu rastgeleliği sabitler.


#### Model Kurulumu ve Eğitim

    lin_reg.fit(X_train, y_train)

Model, eğitim özelliklerini (X_train) ve bunlara karşılık gelen gerçek fiyatları (y_train) kullanarak en uygun düz çizgiyi (regresyon doğrusu) bulur.
    y_pred = lin_reg.predict(X_test)

Eğitilen modeli kullanarak, daha önce görmediği test verilerindeki (X_test) RAM değerleri için fiyat tahmini (y_pred) yapar.

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2_score_test = lin_reg.score(X_test, y_test)

Ortalama karekök hata hesaplanır.Modelin tahminlerinin gerçek değerden ne kadar saptığını hesaplar.

Belirleme katsayısı hesaplanır.Modelin fiyattaki değişim yüzdesini açıklar.

#### Görselleştirme


Grafik boyutu,gerçek veri noktaları ve regresyon doğrusu oluşturulur.

    plt.figure(figsize=(10, 6))
    plt.scatter(X_test, y_test, color='blue', label='Veri Noktaları(Cihazlar)')
    plt.plot(X_test, y_pred, color='red', label='Tahmin Edilen Doğrusal Regresyon')
    plt.title('RAM ve ABD Lansman Fiyatı Arasındaki İlişki')
    plt.xlabel('RAM')
    plt.ylabel('Launched Price (USA)')
    plt.legend()
    plt.show()
![Linear Regresyon](images/linear_regression.png)
### Polinomal Regresyon

RAM özelliğini alır ve degree=2 (ikinci derece/karesel) kullanarak onu yeni bir özellik setine dönüştürür. Basitçe, orijinal  (RAM) değerini x^2 formunda genişletir.

        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=0.2,
            random_state=42
        )

        degree = 2
        poly_features = PolynomialFeatures(degree=degree)
        X_train_poly = poly_features.fit_transform(X_train)
        X_test_poly = poly_features.transform(X_test)
        poly_reg = LinearRegression()
        poly_reg.fit(X_train_poly, y_train)
        y_pred = poly_reg.predict(X_test_poly)

Eğitim özelliklerini (X_train) alır ve onu 1, x, x^2 formatına dönüştürür. fit_transform hem dönüşümü öğrenir hem de uygular.

Test özelliklerini (X_test) alır ve eğitim verisinden öğrenilen kuralları kullanarak aynı formata dönüştürür.

Bu yeni, genişletilmiş özellikler seti üzerine klasik bir Doğrusal Regresyon (LinearRegression) modeli eğitilir. Bu, modelin aslında bir eğri öğrenmesini sağlar.

Grafikte de görüldüğü gibi polinomal regresyon doğru bir yaklaşım değildir.Basitlik ve yorumlanabilirlik için fazla karmaşıktır.

![Polinomal Regresyon](images/polinomial_regression.png)

### Random Forest

Karar ağaçlarından oluşur.Ağaçların ortalamasını alır.

Tahminleri yorumlamak zordur.

        rf_model = RandomForestRegressor(n_estimators=100, random_state=42)

Random forest modelini başlatır. n_estimators=100 parametresi, modelin 100 farklı Decision Tree oluşturacağı anlamına gelir.

        plt.subplot(1, 2, 1)
        sns.scatterplot(x=y_test, y=y_pred, color='green', alpha=0.6, s=70)
        
Test kümesindeki gerçek fiyatları (y_test) ve modelin tahmin ettiği fiyatları (y_pred) mor noktalar olarak çizer. Noktaların dağılımı, modelin performansını gösterir.

        min_val = min(y_test.min(), y_pred.min())
        max_val = max(y_test.max(), y_pred.max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='İdeal Tahmin Çizgisi')

Grafiğe y=x doğrusunu çizer. Tüm noktaların bu çizgi üzerinde toplanması, hatasız tahmin anlamına gelir. Noktalar bu çizgiye ne kadar yakınsa, model o kadar iyidir.
        
Grafikte görüldüğü gibi linear regresyonla arasında çok büyük bir fark yoktur ve bu durumda uygun olan daha basit yorumlaması kolay olan linear regresyonu kulllanmaktır.

![Random Forest ](images/random_forest.png)


### Lojistik Regresyon 

Lojistik regresyon uygulamak için kategorik ifadeler olması gerekir.

Lojistik regresyonun çıktısı 0-1 arasındadır.

Fiyat tahmini için uygun değildir.

### SVR

Amacı, veri noktalarının çoğunu epsilon adı verilen belirli bir hata marjı içinde tutan en iyi sınırı bulmaktır . 

Hata marjı içindeki noktalar umursanmaz, sadece bu marjın dışındaki noktalar (Destek Vektörleri) hataya neden olur.

        scaler_X = StandardScaler()

Veri ölçeklendirme için StandardScaler objesi başlatılır. Bu veriyi ortalaması 0, standart sapması 1 olacak şekilde dönüştürür.

        svr_model = SVR(kernel='rbf', C=1000, epsilon=0.1)

SVR modelini başlatır. Bu parametreler, modelin karmaşıklığını ve hata toleransını belirler: kernel='rbf' ile modelin doğrusal olmayan ilişkileri yakalaması sağlanır.

        svr_model.fit(X_train_scaled, y_train)
        svr_model.fit(X_train_scaled, y_train)
        y_pred = svr_model.predict(X_test_scaled)

Model, ölçeklendirilmiş eğitim verileri üzerinde en uygun hata marjını ($\epsilon$) ve destek vektörlerini bulmayı öğrenir.

Tahminler ölçeklendirilmiş test verisi üzerinden yapılır.


Grafikte görüldüğü gibi düşük açıklama gücü ve yüksek hata payından dolayı uygun değildir.

![SVR](images/SVR.png)

### Çoklu Linear Regresyon 
Çoklu regresyon birden fazla özelliğin bağımlı değişkeni  nasıl etkilediğini bulmayı sağlar.

Korelasyon matrisine baktığımızda RAM den sonra en  güçlü ilişki Mobile Weight ile kurulmuştur.

        target = "Launched Price (USA)"
        features = ["RAM", "Mobile Weight"]

RAM ve Mobile Weight’ın birlikte kullanılması da basit regresyondan farklı bir sonuca ulaşmamıştır.


![Çoklu Linear](images/multiple_linear.png)

## Sonuç

Veri setindeki özellikler için gerekli temizlik yapılarak  sayısal değerler dışındaki ifadeler çıkarılmıştır.

Fiyat tahmini yapmak için en uygun özellik olarak RAM  belirlenmiştir.Isı haritası bunu doğrular.

Regresyon modelleri uygulanmıştır ve en uygun olarak linear regresyon seçilmiştir.

Uygulanan regresyon modellerindeki ortalama hata ve açıklayıcılık oranları genel olarak birbirine yakındır.Bu durumdan dolayı basit ve anlaşılır olması için linear regresyonun seçilmesi en uygunudur.





