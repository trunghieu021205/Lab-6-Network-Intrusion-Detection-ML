Báo cáo EDA và Naive Bayes - TV3



1\. Tổng quan

\- Dataset: CIC-IDS2017 (8 file CSV, 79 cột, \~2.5 triệu dòng)

\- Mục tiêu: Phân tích dữ liệu, đánh giá mô hình Naive Bayes sau khi cân bằng.



2\. Kết quả EDA



2.1 Phân bố tấn công

!\[Attack distribution](eda\_plots/attack\_distribution.png)

\- Nhận xét: Lớp `BENIGN` chiếm phần lớn, các tấn công như `DDoS`, `PortScan` xuất hiện nhiều, nhiều lớp rất ít mẫu → dữ liệu mất cân bằng.



2.2 Ma trận tương quan

!\[Correlation heatmap](eda\_plots/correlation\_heatmap.png)

\- Một số cặp feature có tương quan cao (ví dụ `Flow Duration` và `Tot Fwd Pkts`).



2.3 Cân bằng dữ liệu

!\[Class balance comparison](eda\_plots/class\_balance\_comparison.png)

\- Trước cân bằng: mất cân bằng rõ.

\- Sau cân bằng (SMOTE + RandomUnderSampler): phân bố đều hơn.



3\. Mô hình Naive Bayes



3.1 Thông số

\- Thuật toán: `GaussianNB` (scikit-learn)

\- Dữ liệu: `final\_data.csv` (đã cân bằng, 18 features)



3.2 Kết quả đánh giá

\- Accuracy: 0.52

\- Precision (weighted): 0.78

\- Recall (weighted): 0.52

\- F1-score (weighted): 0.52



Confusion matrix:\*\*

!\[Confusion matrix - Naive Bayes](NaiveBayes\_confusion\_matrix.png)



3.3 Nhận xét

\- Naive Bayes cho kết quả trung bình, đặc biệt kém trên các lớp ít mẫu (như class 14 không dự đoán được).

\- Nguyên nhân: Giả định độc lập giữa các feature không đúng với dữ liệu mạng.



4\. Kết luận

\- Cần cân bằng dữ liệu trước khi train.

\- Naive Bayes không phải mô hình tối ưu cho NIDS; các mô hình mạnh hơn (Random Forest, SVM) sẽ được đánh giá.

