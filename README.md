## Predicting Product Rating Scores Based on Product Reviews



**Dataset**

The dataset used in this assignment contains two columns: `Score` and `Text`. The Score represents the product rating, which ranges from 1 to 5, while the Text column contains the product review written by customers. 

 Class Imbalance
 
A significant class imbalance was evident in the dataset, as follows:
- Score 5.0: Over 30,000 reviews
- Score 4.0: Over 10,000 reviews
- Score 3.0: About 5,000 reviews
- Score 2.0: About 3,000 reviews
- Score 1.0: About 4,000 reviews

 

This class imbalance could bias models toward the dominant class (Score 5.0). After loading the dataset, it was observed that the dataset had 54,985 reviews, out of which 54,947 had valid scores. There were missing values and which were subsequently removed, ensuring clean data for analysis.



Suitability and Problems
The dataset's size and range of scores made it suitable for supervised machine learning. However, class imbalance and potential subjectivity in ratings were challenges. Additionally, the dataset contained unstructured text data, requiring preprocessing to extract meaningful features.


**Data Preparation**

 Preprocessing Steps

A comprehensive text preprocessing pipeline was implemented using the following steps:
1. Lowercasing: All text was converted to lowercase to maintain uniformity.
2. URL Removal: URLs were removed using regular expressions to clean unnecessary information.
3. Special Character Removal: Punctuation, numbers, and special characters were removed to retain only alphabets.
4. Tokenization: Text was split into individual words using the NLTK tokenizer.
5. Stopword Removal: Common words like "the," "is," and "and" were removed using the NLTK stopword list.
6. Rejoining Tokens: Processed tokens were joined back into clean sentences.

These preprocessing steps ensured the text data was clean and suitable for vectorization and further modeling.

*Data Splitting*
 
 
The dataset was divided into training and testing sets using a 80-20 split to ensure sufficient data for both model training and evaluation. The following variables were used:
- X_train, X_test: Cleaned text data
- y_train, y_test: Corresponding product scores

*Text Representation*
 
 
A TextVectorization layer was used to convert text into numerical data for machine learning models. Key parameters included:
- MAX_VOCABULARY_WORDS = 5000: Limited to the most common 5,000 words.
- MAX_SEQUENCE_LENGTH = 200: Each review was truncated or padded to 200 words.
- EMBEDDING_DIM = 10: Word embeddings were generated with a size of 10 dimensions.

This representation allowed models like Naïve Bayes and k-Nearest Neighbors (k-NN) to handle text input efficiently.


**Machine Learning Models**

 i. Naïve Bayes
The Multinomial Naïve Bayes (NB) classifier was used, as it is suitable for text classification tasks. It applies Bayes' theorem and assumes conditional independence between words. Using the vectorized data, the model was trained and evaluated. 

 ii. k-Nearest Neighbors (k-NN)
The k-NN classifier was used to predict scores based on the closest training samples. Various values of k (1, 3, 5, 7, 9) were tested to find the optimal number of neighbors. Accuracy was observed to improve as k increased, with the best performance at k = 9.

 iii. Convolutional Neural Network (CNN)
The CNN model applied multiple convolutional layers with ReLU activation to extract spatial patterns from the text data. Key layers included:
- Embedding Layer: Converted words to dense vectors of size 128.
- Conv1D and MaxPooling: Detected local patterns in the data.
- Flatten and Dense Layers: Performed classification using a softmax activation function.

The CNN demonstrated significant improvements over traditional models.

 iv. Long Short-Term Memory (LSTM)
The LSTM model was designed to capture sequential dependencies using memory cells. The architecture included:
- Embedding Layer: Similar to the CNN.
- Two LSTM Layers: Captured long-term dependencies in the text data.
- Dense Layers: Classified the outputs using softmax activation.

LSTMs are particularly effective for tasks like sentiment analysis where context matters.


**Experimental Results**

The models were evaluated using accuracy, F1-score, precision, and recall. The results are summarized below:

| Model               | Accuracy | F1-Score | Precision | Recall |
|---------------------|----------|----------|-----------|--------|
| Naïve Bayes         | 0.435    | 0.215    | 0.224     | 0.222  |
| k-Nearest Neighbors | 0.473    | 0.178    | 0.202     | 0.201  |
| CNN                 | 0.601    | 0.440    | 0.451     | 0.436  |
| LSTM                | 0.630    | 0.452    | 0.487     | 0.436  |

 Observations:
- LSTM outperformed other models, achieving the highest accuracy of 63.0% and a F1-score of 0.452.
- CNN followed closely with an accuracy of 60.1%. Its ability to capture local patterns in text contributed to its performance.
- Naïve Bayes and k-NN struggled, with F1-scores below 0.22. These traditional models were unable to capture complex linguistic features.

---

**Discussion**

 Model Performance
- LSTMs excelled due to their sequential memory capabilities, making them well-suited for text classification. 
- CNNs effectively extracted local patterns but struggled with long-term dependencies compared to LSTMs.
- Naïve Bayes performed poorly due to its oversimplified assumptions of word independence.
- k-NN is inefficient with large datasets, leading to poor accuracy.

*Class Imbalance Impact*

The significant class imbalance likely caused models to predict dominant classes (Score 5.0) more frequently. This was evident in the high recall for Score 5.0 but low recall for lower scores. Techniques like class weighting could mitigate this effect.

*Limitations*
- The models were only trained for 5 epochs. Additional training could improve performance.

*Further Improvements*
- Fine-tuning hyperparameters, increasing epochs, and using larger embedding dimensions may yield better results.
- Implementing ensemble models that combine CNN and LSTM could leverage both models' strengths.

**Conclusion**

The LSTM model proved to be the most effective for predicting product ratings from reviews, demonstrating superior accuracy and F1-scores and CNN was a close second.

