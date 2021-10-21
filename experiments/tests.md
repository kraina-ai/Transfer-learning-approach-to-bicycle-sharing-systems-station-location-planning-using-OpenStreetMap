Single hex embedding methods:
- Base category tf-idf (count) vector
- Number of points per category + lines length inside per category + polygons area inside per category
- Prevoius + tfidf per category
  - sustenance: amenity tfidf
  - education: amenity tfidf
  - transport: amenity or public_transport
  - finance: amenity tfidf
  - healthcare: amenity tfidf
  - culture_art_entertainment: amenity
  - other: amenity tfidf (place_of_worship + religion)
  - historic: historic tfidf
  - leisure: leisure tfidf
  - shops: shop tfidf
  - sport: sport tfidf
  - tourism: tourism tfidf
 - Previous + autoencoder to reduce dimentionality (nn in tensorflow)

Neighbors embedding methods:
  - Average from given level and concatenate
  - Average from given level and average all levels
  - Average from given level and average with diminishing weights (1/n, 1/n^2)

Transfer learning:
  - Single city only
  - Learn on multiple cities and validate on them
  - Learn on one city and validate on another
  - Learn on multiple cities and validate on multiple different

Metrics:
  - Accuracy
  - F1-Score
  - Custom: 1 if TP, 1 if TN, 0 if FN, 1/(distance to closest hex) if FP

Train-Test inbalance:
  - 1:1 ratio
  - 2:1 non-station ratio
  - 3:1 non-station ratio
  - 5:1 non-station ratio
  <!-- - 10:1 non-station ratio -->

Classifiers:
  - kNN
  - RBF SVM
  - Random Forest
  - AdaBoost
  - NLP (custom)