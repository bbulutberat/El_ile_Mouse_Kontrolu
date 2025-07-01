## El Hareketleriyle Mouse Kontrolü
- Bu proje, webcam kullanarak işaret parmağı hareketlerini takip ederek fare imlecini kontrol etmeyi sağlar. OpenCV, MediaPipe ve PyAutoGUI kütüphaneleri ile geliştirilmiştir. Optik akış (Lucas-Kanade yöntemi) kullanılarak işaret parmağının hareketi izlenir ve ekran koordinatlarına dönüştürülerek fare imleci hareket ettirilir.

## Optik Akış
- Optik akış, bir videoda iki ardışık kare arasında piksel başına hareket tahminidir. Temel olarak, Optik Akış, iki komşu görüntü arasındaki bir nesnenin yer değiştirme farkı olarak piksel için kaydırma vektörünün hesaplanmasını ifade eder. Optik Akışın ana fikri, nesnenin hareketi veya kamera hareketlerinin neden olduğu yer değiştirme vektörünü tahmin etmektir.

- Birinci karedeki piksel I(x, y, t) olarak tanımlanır. Burada x ve y piksel konumu t ise iki kare arasında geçen süredir.
- Bir nesnenin hareket ediyor olsa bile görüntüdeki renginin ve yoğunluğunun değişmediği varsayılarak piksel hareket ederken ikinci karedeki görüntüsü taylor açılımıyla I(x + dx, y + dy, t +dt) şeklinde hesaplanır.

**Lucas-Kanade**
- İki tür Optik Akış vardır ve ilki Seyrek Optik Akış olarak adlandırılır. Belirli nesne kümesi için hareket vektörünü hesaplar (örneğin, görüntüde algılanan köşeler). Bu nedenle, Optik Akış hesaplamasının temelini oluşturacak olan görüntüden özellikleri çıkarmak için bazı ön işlemler gerekir. Lucas-Kanade, OpenCV’de, yukarıda bahsettiğimiz taylor açılımlarından sonra oluşan iki bilinmeyenli denklemi çözmek için kullanılan bir algoritmadır. Bu yöntemin ana fikri, yakındaki piksellerin aynı yer değiştirme yönüne sahip olduğu yerel bir hareket sabitliği varsayımına dayanmaktadır. Bu varsayım, iki değişkenli denklem için yaklaşık çözümü elde etmeye yardımcı olur. 

**Avantajları**
- Piksellerdeki küçük ve sürekli hareketler için yüksek doğruluk sağlar.
- Hesaplama hızı olarak verimlidir.
- OpenCV gibi kütüphanelerde optimize edilmiş haliyle bulunur, doğrudan kullanılabilir.

**Dezavantajları**
- Büyük hareketleri doğru şekilde tahmin edemez; hareket çok fazlaysa akış bozulur.
- Görüntüdeki parlaklık (illumination) değişimleri, tahmin hatalarına yol açabilir.
- Gürültülü görüntülerde hatalı yön ve hız tahmini yapılabilir.
- Döndürme, şekil değiştirme (deformasyon) veya perspektif değişimi gibi karmaşık durumlarda başarı oranı düşer.

## Mediapipe 
- MediaPipe Hands, gerçek zamanlı olarak elleri algılayıp, her elde 21 adet anahtar nokta (landmark) tespiti yapabilen bir modeldir.
- Elin yönü, açısı, parmak duruşları ve parmak uçları gibi detaylar landmark koordinatları sayesinde yüksek doğrulukla analiz edilebilir.
- MediaPipe, derin öğrenme tabanlı bir modelle çalışır.
- Model, tek karede hem el algılama hem de landmark tahmini yaparak hızlı ve doğru sonuçlar sunar

## Nasıl Çalışır? 
- ```cv2.VideoCapture(0)``` ile kamera başlatılır.
- ```mp.solutions.hands.Hands()``` ile yalnızca bir el tespit edilecek şekilde el modeli yüklenir.
- Kamera görüntüsü yatay eksende çevrilir ```(cv2.flip)```.
- MediaPipe modeli ile görüntüdeki el ve parmak landmark’ları tespit edilir.
- 8 numaralı landmark (işaret parmağı ucu) alınır.
- İşaret parmağının koordinatları alınarak ```p0``` olarak belirlenir.
- ```optic_flow()``` fonksiyonu çağrılır ve Lucas-Kanade Optical Flow ile hareket takibi başlar.
- Her karede ```cv2.calcOpticalFlowPyrLK()``` ile işaret parmağının eski ve yeni pozisyonu (p0 → p1) karşılaştırılır.
- Hareket yönü bir çizgi ile çizilir.
- Yeni pozisyona göre fare hareket ettirilir ```pyautogui.moveTo()```
- Parmağın görüntüdeki konumu, ekran çözünürlüğüne oranlanarak ekran koordinatlarına dönüştürülür.
- İmleç ekran üzerinde taşınır.
- ESC tuşuna basıldığında kamera kapanır ve tüm OpenCV pencereleri temizlenir.

## KOD AÇIKLAMASI

``` 
import cv2
import mediapipe as mp
import numpy as np
import pyautogui

class MouseControl():

    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        # Mediapipe el tespit modeli yüklenir.
        mp_hands = mp.solutions.hands
        # El tespiti için gerekli parametreler verilir.
                                    # Elin hareketi takip mi edilecek yoksa her seferinde yeniden mi tespit edilecek.
        self.hands = mp_hands.Hands(static_image_mode=False,
                                     max_num_hands=1,   # Maksimum tespit edilecek el sayısı.
                                     min_detection_confidence=0.5, # El algılamanın başarılı olması için min güven puanı.
                                     min_tracking_confidence=0.5) # El takibinin başarılı olması için min güven puanı.

        
    def run(self):
        while True:
            # Her frame döngü içinde tek tek alınır.
            ret, frame = self.cap.read()
            if not ret:
                break
            # Görüntü yatay eksende çevirilir.
            frame = cv2.flip(frame, 1)
            # Mediapipe rgb görüntü istediği için frame rgb formatına çevirilir.
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Algılama işlemi 
            results = self.hands.process(rgb)
            
            # Görüntü ekranda gösterilir.
            cv2.imshow("MouseKontrol", frame)
            key = cv2.waitKey(30)

            # Eğer el algılandıysa
            if results.multi_hand_landmarks:
                # Algılanan landmarklar tek tek for döngüsünde işlenir
                for hand_landmarks in results.multi_hand_landmarks:
                    # Mediapipe koordinatları ölçek olarak verdiği için görüntü boyutu ile normalizasyon yapılır.
                    h, w, _ = frame.shape
                    # İşaret parmağının ucunu algılayan landmark seçilir.
                    lm = hand_landmarks.landmark[8]
                    # Algılanan noktanın koordinatları hesaplanır.  
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    # Lucas-kanade için koordinatlar uygun formata getirilir.
                    p0 = np.array([[cx, cy]], dtype=np.float32).reshape(-1, 1, 2)
                    # Lucas-kanade için görüntü griye çevirilir.
                    prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                    self.optic_flow(p0, prev_gray, frame)
                    break   

            if key == 27: # ESC tuşuna basılırsa döngüden çık
                break
        self.cap.release()
        cv2.destroyAllWindows()

    def optic_flow(self, p0, prev_gray, frame):
        #Lukas-kanade için parametreler ayarlanır.
        lk_params = dict(winSize  = (15, 15), # Hareketi hesaplamak için her piksel çevresinde oluşturulacak pencere boyutu
                         maxLevel = 2, # Akış hesaplaması için görüntünün oluşturulan Gaussian piramidindeki maksimum seviye sayısı.
                     # Optik akış iterasyonu, ya maksimum 10 adımda tamamlanır ya da hareketin değişimi 0.03’ten küçük olduğunda durur.
                         criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)) 
        
        # Hareketin görselleştirilmesi için aynı boyutta maske oluşturulur.
        mask = np.zeros_like(frame)  

        while True:
            # Sıradaki frame alınır
            ret, next_frame = self.cap.read()
            if not ret:
                break

            next_frame = cv2.flip(next_frame, 1)
            next_gray = cv2.cvtColor(next_frame, cv2.COLOR_BGR2GRAY)

            # Lucas-kanade ile optik akış hesaplanır. Noktanın diğer framedeki koordinatları p1 değişkeninde saklanır.
            p1, st, err = cv2.calcOpticalFlowPyrLK(prev_gray, next_gray, p0, None, **lk_params)

            if p1 is not None :
                # p0 ve p1 koordinatlarını numpy dizisinden tek boyutlu diziye çevrilir.
                a, b = p0.ravel()
                c, d = p1.ravel()
                # Maskeye hareket gösterimi için çizim yapılır.
                mask = cv2.line(mask, (int(a), int(b)), (int(c), int(d)), (0,255,0), 2)
                # Hareket sonundaki nokta yuvarlak ile işaretlenir.
                next_frame = cv2.circle(next_frame, (int(c), int(d)), 5, (0,255,0), -1)
                self.mousemove(c, d, next_frame)

                # Maskeye çizilen hareket gösterimini ekranda göster için frame ile birleştirilir.
                img = cv2.add(next_frame, mask)

                cv2.imshow('MouseKontrol', img)
                k = cv2.waitKey(30)
                # Tekrar optik akış hesaplaması için son frame eski frame atanır ve sıradaki frame'i almak için döngü başa döner.
                prev_gray = next_gray.copy()
                # Lucask-kanade için yeni p0 noktası uygun formata çevirilir.
                p0 = p1.reshape(-1, 1, 2)

            if k == 27:
                    break

    def mousemove(self, x, y, frame):
        # Mouse kontolü için ekran boyutu alınır.
        screen_w, screen_h = pyautogui.size()
        # Pencere boyutu alınır. 
        frame_h, frame_w = frame.shape[:2]
        # Mouse hareket koordinatları pencere ve ekran boyutuna göre ayarlanır.
        move_x = int(x * (screen_w / frame_w))
        move_y = int(y * (screen_h / frame_h))
        # Mouse belirlenen koordinatlara hareket ettirilir.
        pyautogui.moveTo(move_x, move_y)

if __name__ == "__main__":
    mc = MouseControl()
    mc.run()
``` 


