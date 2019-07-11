# DSCI 563: Unsupervised Learning

How to find groups and other structure in unlabeled, possibly high dimensional data. Dimension reduction for visualization and data analysis. Clustering and model fitting via the EM algorithm.

## Lectures


| # |     Date      | Day |  Topic  |  Slides  |
|---|---------------|-----|---------|----------|
| 1 | 2019-01-03   | Thur | Unsupervised paradigm; Clustering: K-Means and FCM; Choosing K;  | [lecture 1](https://github.ubc.ca/MDS-2018-19/DSCI_563_unsup-learn_students/blob/master/lectures/Lecture1.ipynb)  | 
| 2 | 2019-01-08   | Tue  | K-Medians; K-Medoids; Hierarchical Clustering; DBSCAN  | [lecture 2](https://github.ubc.ca/MDS-2018-19/DSCI_563_unsup-learn_students/blob/master/lectures/Lecture2.ipynb)  |
| 3 | 2019-01-10   | Thur | EM algorithm and Gaussian Mixtures  | [lecture 3](https://github.ubc.ca/MDS-2018-19/DSCI_563_unsup-learn_students/blob/master/lectures/Lecture3.ipynb)  |
| 4 | 2019-01-15   | Tue  | Principal Component Analysis ([Mike's class](https://www.youtube.com/watch?v=7cBkOC_UD4o&list=PLWmXHcz_53Q02ZLeAxigki1JZFfCO6M-b&index=25&t=0s)) | [lecture 4](https://github.ubc.ca/MDS-2018-19/DSCI_563_unsup-learn_students/blob/master/lectures/Lecture4.ipynb)  |
| 5 | 2019-01-17   | Thur | NMF, Sparse PCA ([Mike's class](https://www.youtube.com/watch?v=ghLOWBlzWyw&t=2221s))  | [lecture 5](https://github.ubc.ca/MDS-2018-19/DSCI_563_unsup-learn_students/blob/master/lectures/lecture5.ipynb)  |
| 6 | 2019-01-22   | Tue  | Recommender Systems ([Mike's class](https://www.youtube.com/watch?v=mBFChbO-SNI)) | [lecture 6](https://github.ubc.ca/MDS-2018-19/DSCI_563_unsup-learn_students/blob/master/lectures/lecture6.ipynb)  |
| 7 | 2019-01-24   | Thur | More on Recommender Systems and Multidimensional Scaling ([Mike's class](https://www.youtube.com/watch?v=rR9kLt8hxq0&index=27&list=PLWmXHcz_53Q02ZLeAxigki1JZFfCO6M-b))  |   |
| 8 | 2019-01-29   | Tue  | GAP statistic, FCM, t-SNE, Isomap, eigenfaces  |   |
 

## Reference Material and other Resources
* **[JWHT13]**: James, G., Witten, D., Hastie, T. and Tibshirani, R.
An Introduction to Statistical Learning. 2013. Springer-Verlag New York
	- [Book page](http://www-bcf.usc.edu/~gareth/ISL/), [Book PDF](http://www-bcf.usc.edu/~gareth/ISL/ISLR%20Sixth%20Printing.pdf)

* **[HTF09]**: Hastie, T., Tibshirani, R. and Friedman, J.
The Elements of Statistical Learning. 2009. Second Edition. Springer-Verlag New York
	- [Book page](https://web.stanford.edu/~hastie/ElemStatLearn/), [Book PDF](https://web.stanford.edu/~hastie/ElemStatLearn/printings/ESLII_print12.pdf)

- Week 1:
	* JWHT13: 10.3; HTF09: 13.2, 14.3
	* A **robust** K-means algorithm: [Paper](http://dx.doi.org/10.18637/jss.v047.i12), [CRAN](https://cran.r-project.org/package=tclust)
	* A **robust and sparse** K-means algorithm: [Paper](http://dx.doi.org/10.18637/jss.v072.i05), [CRAN](https://cran.r-project.org/package=RSKC)
	* Validating the number of clusters via classification: [Paper](http://dx.doi.org/10.1186/gb-2002-3-7-research0036)
- Week 2:
	* [Mixture models notes](http://www.cs.toronto.edu/~rgrosse/csc321/mixture_models.pdf) from U. Toronto CSC 321
	* [Understanding mixture models and expectation-maximization (using baseball statistics)](http://varianceexplained.org/r/mixture-models-baseball/)
	* Model based clustering: Section 13.2 and 14.3 from HTF09
	* EM algorithm: Section 8.5 from HTF09
	* [Information Theory, Inference, and Learning Algorithms](http://www.inference.org.uk/itprnn/book.pdf), Chapters 20-22
- Week 3:
	8 [PCA explained visually](http://setosa.io/ev/principal-component-analysis/)
	* Principal Components Analysis: Section 14.5 from HTF09
	* Factor Analysis: Section 14.7 from HTF09
- Week 4:
	* Multidimensional Analysis: Section 14.8 from HTF09

## Linear algebra review

There are a bunch of suggestions [here](https://ubc-mds.github.io/resources_pages/prep_moocs/). We particularly recommend [essence of linear algebra](https://www.youtube.com/watch?v=kjBOesZCoqc&list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab) (YouTube series) and
[Immersive linear algebra](http://immersivemath.com/ila/index.html) (interactive e-book).
