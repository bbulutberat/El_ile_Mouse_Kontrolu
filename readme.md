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
import numpy as np
import mediapipe as mp
import pyautogui

class MouseControl():
    
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        # Mediapipe el tespit modeli yüklenir.
        mp_hands = mp.solutions.hands
        # El tespiti için gerekli parametreler verilir.
        self.hands = mp_hands.Hands(static_image_mode=False,
                                     max_num_hands=1,
                                     min_detection_confidence=0.5,
                                     min_tracking_confidence=0.5)
        # Lukas-kanade için parametreler ayarlanır.
        self.lk_params = dict(winSize  = (15, 15), 
                         maxLevel = 2, 
                         criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
        # El tespiti için flag
        self.mp_control = False
        # Optik Akış flagi
        self.p0 = None
        

    def main(self):
        
        while True:
            # Frame alınır
            ret, frame = self.cap.read()
            if not ret:
                break
            # Dikey eksende döndürülür.
            frame = cv2.flip(frame, 1)

            if self.mp_control == False:
                # Mediapipe için rgb formata çevirilir.
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = self.hands.process(rgb)
                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        h, w, _ = frame.shape
                        # İşaret parmağı koordinatları tespit edilir
                        lm = hand_landmarks.landmark[8]  
                        cx, cy = int(lm.x * w), int(lm.y * h)
                        self.p0 = np.array([[cx, cy]], dtype=np.float32).reshape(-1, 1, 2)
                        # Görüntü optik akış için griye çevilir.
                        self.frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                        # Çizimler için maske oluşturulur
                        self.mask = np.zeros_like(frame)
                        # Tekrar el tespiti yapılmamaı için bayrak değeri True yapılır.
                        self.mp_control = True
            
            # p0 bulundu iste optik flow fonksiyonuna git
            if self.p0 is not None:
                img = self.optic_flow()
            

            else:
                img = frame

            cv2.imshow("MouseControl", img)
            key = cv2.waitKey(1)
            if key == 27:
                break
            
    def optic_flow(self):
        # Optik akış için sıradaki frame alınır
        ret, frame2 = self.cap.read()
        frame2 = cv2.flip(frame2, 1)
        frame2_gray = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        
        # Optik akış hesaplaması yapılır.
        p1, st, err = cv2.calcOpticalFlowPyrLK(self.frame_gray, frame2_gray, self.p0, None, **self.lk_params)

        if p1 is not None:
            # Koordinatlar uygun formata çevirilir.
            a, b = self.p0.ravel()
            c, d = p1.ravel()
            # Maskeye hareket çizimleri yapılır
            self.mask = cv2.line(self.mask, (int(a), int(b)), (int(c), int(d)), (0,255,0), 2)
            # Hareket edilen nokta kapalı daire ile işaretlenir.
            self.frame2 = cv2.circle(frame2, (int(c), int(d)), 5, (0,255,0), -1)
            # maske ve frame birleştirilerek çizim işlemi tamamlanır.
            img = cv2.add(frame2, self.mask)
            # Mouse hareketleri için mousemove fonksiyonu çağırılır.
            self.mousemove(c, d, frame2)
            # Optik akış hesaplamasının devamı için önceki görüntü değiştirilir.
            self.frame_gray = frame2_gray.copy()
            # p0 noktası güncellenir.
            self.p0 = p1.reshape(-1, 1, 2)
            return img
    
    def mousemove(self, x, y, frame2):
        # Mouse kontolü için ekran boyutu alınır.
        screen_w, screen_h = pyautogui.size()
        # Pencere boyutu alınır. 
        frame_h, frame_w = frame2.shape[:2]
        # Mouse hareket koordinatları pencere ve ekran boyutuna göre ayarlanır.
        move_x = int(x * (screen_w / frame_w))
        move_y = int(y * (screen_h / frame_h))
         # Mouse belirlenen koordinatlara hareket ettirilir.
        pyautogui.moveTo(move_x, move_y)


if __name__ == "__main__":
    mc = MouseControl()
    mc.main()
``` 


