# **Semantic Analysis with LSA and LDIA**

**Introduction and Overview**
- In Natural Language Processing, Semantic Analysis is the process of relating the Syntactic structures from the levels of Phrases, Clauses, Sentences and Paragraphs to the level of the writing as a whole and to their Language dependent meanings. Latent Semantic analysis (LSA) is a technique in Natural Language Processing, in particular Distributional Semantics, of analyzing relationships between a set of documents and the terms they contain by producing a set of concepts related to the documents and terms. LSA assumes that words that are close in meaning will occur in similar pieces of text. A matrix containing Word counts per Document is constructed from a large piece of Text and a Mathematical technique called Singular Value Decomposition (SVD) is used to reduce the number of rows while preserving the similarity structure among columns. In Natural Language Processing, the Latent Dirichlet Allocation (LDA) is a generative statistical model that allows sets of observations to be explained by unobserved groups that explain why some parts of the Data are similar. LDA is an example of a Topic model and belongs to the machine learning toolbox and in wider sense to the Artificial Intelligence Toolbox.

**Libraries and Dependencies**

```javascript
import numpy as np
import pandas as pd
import nltk                                                                          
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import PCA                                                
from sklearn.decomposition import TruncatedSVD                                       
from nltk.tokenize.casual import casual_tokenize
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import LatentDirichletAllocation as LDiA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
```

**PCA on 3D Vectors**

```javascript
pca = PCA(n_components=2)
clouddf = pd.DataFrame(pca.fit_transform(cloud), columns=list("xy"))  
clouddf.plot(kind="scatter", x="x", y="y")
plt.show()
```

![Image](https://github.com/ThinamXx/SemanticAnalysis__LDA..LDIA/blob/master/Images/Horse.PNG)

**LDA Classifier using LDIA Topic Vectors**
- Latent Dirichlet Allocation (LDIA) is a generative statistical model that allows sets of observations to be explained by unobserved groups that explain why some parts of the Data are similar. LDA is an example of a Topic model and belongs to the machine learning toolbox and in wider sense to the Artificial Intelligence Toolbox. Latent Dirichlet Allocation (LDIA) works with raw Bag of Words (BOW) Count Vectors rather than the Normalized Term Frequency Inverse Document Frequency (TFIDF) Vectors. The Code Snippets mentioned below is the Implementation of LDA Classifier using LDIA Topic Vectors.

```javascript
counter = CountVectorizer(tokenizer=casual_tokenize)
bow_docs = pd.DataFrame(counter.fit_transform(raw_documents=sms.text).toarray(), index=index)
ldia = LDiA(n_components=16, learning_method="batch")
ldia = ldia.fit(bow_docs)
ldia16_topic_vectors = ldia.transform(bow_docs)
ldia16_topic_vectors = pd.DataFrame(ldia16_topic_vectors, index=index, columns=columns)
X_train, X_test, y_train, y_test = train_test_split(ldia16_topic_vectors, sms.spam, test_size=0.3, random_state=1)
lda = LDA(n_components=1)
lda = lda.fit(X_train, y_train)
round(float(lda.score(X_test, y_test)), 2)
```

**LDA Classifier using TFIDF Vectors**
- Term Frequency Inverse Document Frequency (TFIDF) is a Numerical Statistic that is intended to reflect how important a word is to a document in a collection or corpus. It is often used as a weighting factor in searches of Information Retrieval, Text Mining and User Modeling. TFIDF is oftern used by seach engines. The Code Snippets mentioned below is the Implementation of LDA Classifier using TFIDF Count Vectors.

```javascript
tfidf_model = TfidfVectorizer(tokenizer=casual_tokenize)      
tfidf = tfidf_model.fit_transform(raw_documents=sms.text).toarray()
tfidf_docs = tfidf_docs - tfidf_docs.mean(axis=0) 
X_train, X_test, y_train, y_test = train_test_split(tfidf_docs, sms.spam.values , test_size=0.3, random_state=1)
lda = LDA(n_components=1)
lda = lda.fit(X_train, y_train)
round(float(lda.score(X_test, y_test)), 2)
```

**LDA Classifier using LSA Topic Vectors**
- Latent Semantic analysis (LSA) is a technique in Natural Language Processing, in particular Distributional Semantics, of analyzing relationships between a set of documents and the terms they contain by producing a set of concepts related to the documents and terms. LSA assumes that words that are close in meaning will occur in similar pieces of text. A matrix containing Word counts per Document is constructed from a large piece of Text and a Mathematical technique called Singular Value Decomposition (SVD) is used to reduce the number of rows while preserving the similarity structure among columns. The Code Snippets mentioned below is the Implementation of LDA Classifier using LSA Topic Vectors. LSA is one of the most used Algorithmn in search engines. And the Accuracy obtained by LSA Topic Vectors while Implementing with LDA Classifier is the best among all.U

```javascript
pca = PCA(n_components=16)                                                            
pca = pca.fit(tfidf_docs)                                                             
pca_topic_vectors = pca.transform(tfidf_docs)
X_train, X_test, y_train, y_test = train_test_split(pca_topic_vectors.values, sms.spam, test_size=0.3, random_state=1)
lda = LDA(n_components=1)                                                                         
lda = lda.fit(X_train, y_train)
display(round(float(lda.score(X_test, y_test)), 3))
```
