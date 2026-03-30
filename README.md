KORU (Kovaryans Odaklı Risk Uyarı) Sistemi sistemimiz şimdi Github'da 💯 



Günümüz uzay serüvenleri ve çalışmaları için kullanılan uydular, uzay istasyonları ve keşif araçlarının günümüzdeki en büyük problemlerinden birisi uzay çöplüğüdür. Nitekim günümüzde 140 milyondan fazla mikro boyutlu cisim ve 34 binden fazla 10 cm üzeri cisim uzay çöplüğünü oluşturuyor ve uzay araçlarına zarar vererek milyonlarca dolarlık maddi kayıplara yol açıyor. İşte sistemimiz tam olarak burada devreye giriyor.



Sistem, 2021 yılından bu yana olan süreç içerisinde takip edilen uzay çöplerinin verilerini bir arayüz ortamında sunarak hareketlerini simüle etmeyi amaçlar. Bu veriler için uzay çöp havuzlarının verileri TLE formatında toplanarak temizlendi ve model eğitimi için hazır hale getirildi. Model eğitiminde Python dili ve sklearn, numpy, pandas, matplotlib, seaborn ve xgboost kütüphaneleri kullanıldı. Eğitilen model gerçek zamanlı olarak uydu ve uzay çöpü görsellerini oluşturur ve bunları arayüze aktarır.



Arayüz ortamı HTML ve CSS dilleri kullanılarak tasarlandı. Arayüz gerçek zamanlı olarak dünya saati ve eksen hareketleri baz alınarak hesaplandı.  Sol panel uydulardan, sağ panel ise uydulara basıldığında bu uyduların eksenindeki çöplerin isimlerinden oluşuyor. Çarpışmayı göster tuşuna basarak uydu ile çöpün çarpışma ihtimali bulunan konuma hareketleri simüle edilir. Bu simülasyon hızı ayarlanabilir ve gerektiğinde durdurulabilir. Eğer uydu ve çöpün çarpışma durumu bulunursa sistem uyarı vererek uydunun yapması gereken hareketin miktarı ve yönelme miktarı en az yakıt harcaması gerçekleştirilecek şekilde hesaplanır. Ardından bu uyarı ana merkeze gönderilebilir. Merkez panelde simülasyon görseli bulunur ve bütün animasyon ve görüntüler burada oluşur. 📸 
