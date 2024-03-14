
![](lectures/img/563_banner.png)

## Important links 

- [Course Jupyter book](https://pages.github.ubc.ca/mds-2023-24/DSCI_563_unsup-learn_students/README.html)
- [Course GitHub page](https://github.ubc.ca/MDS-2023-24/DSCI_563_unsup-learn_students)
- [Slack Channel](https://ubc-mds.slack.com/messages/563_unsup-learn)
- [Canvas](https://canvas.ubc.ca/courses/130310)
- [Gradescope](https://www.gradescope.ca/courses/12077)
- [YouTube videos](https://www.youtube.com/playlist?list=PLHofvQE1VlGtZoAULxcHb7lOsMved0CuM)
- [Class + office hours calendar](https://ubc-mds.github.io/calendar/)
## Course learning outcomes    
This course is about identifying underlying structure in data. We will talk about clustering, data representation (e.g., dimensionality reduction and word embeddings), and recommendation systems. 

<details>
  <summary>Click to expand!</summary>  
    
By the end of the course, students are expected to be able to
    
- Explain the unsupervised paradigm. 
- Explain the intuition behind clustering and use appropriate clustering algorithms for applications such as image clustering and document clustering. 
- Interpret the results obtained after applying clustering. 
- Explain the intuition behind dimensionality reduction. 
- Broadly explain and use linear dimensionality reduction techniques such as PCA, LSA, and NMF. 
- Explain the intuition of word2vec model to create word embeddings. 
- Train your own word embeddings and use pre-trained word embeddings.
- Explain the recommender systems problem. 
- Broadly explain and use two common approaches to recommender systems: collaborative filtering and content-based filtering. 
- Explain consequences of using recommender systems.  
</details>


## Deliverables

<details>
  <summary>Click to expand!</summary>
    
The following deliverables will determine your course grade:

| Assessment           | Weight  | Where to submit|
| :---:                | :---:   |:---:  | 
| Lab Assignment 1     | 12%     | [Gradescope](https://www.gradescope.ca/courses/12077) |
| Lab Assignment 2     | 12%     | [Gradescope](https://www.gradescope.ca/courses/12077) 
| Lab Assignment 3     | 12%     | [Gradescope](https://www.gradescope.ca/courses/12077) |
| Lab Assignment 4     | 12%     | [Gradescope](https://www.gradescope.ca/courses/12077) |
| Class participation  |  2%     | [iClicker Cloud]() |
| Quiz 1               | 25%     | [Canvas](https://canvas.ubc.ca/courses/106525)     |
| Quiz 2               | 25%     | [Canvas](https://canvas.ubc.ca/courses/106525)     |

See [Calendar](https://ubc-mds.github.io/calendar/) for the due dates. 
</details>

## Teaching team
<details>
  <summary>Click to expand!</summary>
    
| Role | Name  | 
| :------: | :---: |
| Lecture instructor | Varada Kolhatkar |
| Lab instructor | Varada Kolhatkar |
| Teaching assistant | Ngoc Bui|
| Teaching assistant | Mohit Pandey |
| Teaching assistant | Negar Sadrzadeh |
| Teaching assistant | Jordan Yu |
    
</details>  

## Lectures 

### Format
<details>
  <summary>Click to expand!</summary>

This class will follow a semi-flipped classroom format. For four out of the eight lectures, you will be required to watch a few pre-recorded videos (~30 to ~50 min long) before the lecture. All videos are available on YouTube and are linked in the Lecture Schedule below. During lectures, I'll summarize the content from videos but I'll assume that you understand the basic concepts from the videos and we will focus on more advanced material, iClicker exercises, discussions, demos, and class activities. It's optional but highly recommended to download the appropriate datasets provided below and put them under your local `lectures/data` directory, and run the lecture Jupyter notebooks on your own and experiment with the code. 
</details>

### Lecture Schedule

This course occurs during **Block 5** in the 2021/22 school year. 

| Lecture  | Topic  | Assigned videos  | Resources and optional readings |
|-------|------------|-----------|-----------|
| 0     | [Course Information](lectures/00_course-information.ipynb) | | |
| 1     | [K-Means and intro to GMMs](lectures/01_lecture-k-means.ipynb)  | 📹  <li> Videos: [14.1](https://youtu.be/caAuUAXwpb8), [14.2](https://youtu.be/s6AvSZ1_l7I),[14.3](https://youtu.be/M5ilrhcL0oY)| <li>[`sklearn` clustering documentation](https://scikit-learn.org/stable/modules/clustering.html)</li><li>["Spaghetti Sauce" talk by Malcom Gladwell](https://www.ted.com/talks/malcolm_gladwell_on_spaghetti_sauce?language=en)</li><li>[Visualizing-k-means-clustering](https://www.naftaliharris.com/blog/visualizing-k-means-clustering/)</li><li>[Visualizing K-Means algorithm with D3.js](http://tech.nitoyon.com/en/blog/2013/11/07/k-means/)</li><li>[Clustering with Scikit with GIFs](https://dashee87.github.io/data%20science/general/Clustering-with-Scikit-with-GIFs/)</li>|
| 2    | [DBSCAN and Hierarchical Clustering](lectures/02_DBSCAN-hierarchical.ipynb)  | 📹  <li> Videos: [15.1](https://youtu.be/1ZwITQyWpkY), [15.2](https://youtu.be/T4NLsrUaRtg), [15.3](https://youtu.be/NM8lFKFZ2IU) | <li>Comparison of [sklearn clustering algorithms](https://scikit-learn.org/stable/modules/clustering.html#overview-of-clustering-methods)</li><li>[DBSCAN Visualization](https://www.naftaliharris.com/blog/visualizing-dbscan-clustering/)</li><li>[Clustering with Scikit with GIFs](https://dashee87.github.io/data%20science/general/Clustering-with-Scikit-with-GIFs/)</li> | 
| 3    | [Dimensionality Reduction Intro](lectures/03_PCA-intro.ipynb) | 📹  <li> Videos: [17.1](https://youtu.be/r-DwXpg1YDI), [17.2](https://youtu.be/33TRSSuzALw), [17.3](https://youtu.be/g5w3o1TE6hU)</li> | <li>[PCA visualization](https://setosa.io/ev/principal-component-analysis/)</li><li>[Introduction to Machine Learning with Python book Chapter 3](https://learning.oreilly.com/library/view/introduction-to-machine/9781449369880/ch03.html)</li><li>[Mike's PCA video from CPSC 340](https://www.youtube.com/watch?v=7cBkOC_UD4o&list=PLWmXHcz_53Q02ZLeAxigki1JZFfCO6M-b&index=25&t=0s)</li><li>[StatQuest PCA video](https://www.youtube.com/watch?v=FgakZw6K1QQ&feature=youtu.be)</li> |
| 4    | [More PCA, LSA, NMF, Autoencoders](lectures/04_LSA-NMF-AE.ipynb) | No videos | 
|   5   | [Word Vectors, Word Embeddings](lectures/05_word-embeddings.ipynb) | 📹  <li> Videos: [18.1](https://youtu.be/7nGGogNUrtg), [18.2](https://youtu.be/aj8OWol-H2I), [18.3](https://youtu.be/rWoA-IKGDa8)</li> | Word2Vec papers: <li>[Distributed representations of words and phrases and their compositionality](https://papers.nips.cc/paper/5021-distributed-representations-of-words-and-phrases-and-their-compositionality.pdf)</li> <li>[Efficient estimation of word representations in vector space](https://arxiv.org/pdf/1301.3781.pdf)</li> <li>[word2vec Explained](https://arxiv.org/pdf/1402.3722.pdf)</li><li>[Debiasing Word Embeddings](http://papers.nips.cc/paper/6228-man-is-to-computer-programmer-as-woman-is-to-homemaker-debiasing-word-embeddings.pdf)</li>|
|   6   | [Using Word Embeddings, Manifold Learning](lectures/06_more-word2vec-tsne.ipynb) | No videos | <li>[t-SNE tutorial](https://github.com/oreillymedia/t-SNE-tutorial)</li><li>[How to use t-SNE effectively](https://distill.pub/2016/misread-tsne/)</li><li>[LargeVis](https://github.com/elbamos/largeVis)</li><li>[UMAP](https://github.com/lmcinnes/umap)</li> |
| 7    | Recommender Systems I | No videos | <li>[Collaborative filtering for recommendation systems in Python, by N. Hug](https://www.youtube.com/watch?v=z0dx-YckFko)</li><li>[How Netflix’s Recommendations System Works](https://help.netflix.com/en/node/100639)</li>|
| 8    | Recommender Systems II | No videos | <li>[SVDfeature](https://www.jmlr.org/papers/v13/chen12a.html)</li>|



### Datasets
Here is the list of [Kaggle](https://www.kaggle.com/) datasets we'll use in the lectures. 
- A small subset of [200 Bird Species with 11,788 Images](https://www.kaggle.com/datasets/veeralakrishna/200-bird-species-with-11788-images) (available [here](https://github.ubc.ca/mds-2021-22/datasets/blob/master/data/birds.zip))
- A tiny subset of [Food-101](https://www.kaggle.com/datasets/kmader/food41?select=food_c101_n10099_r32x32x1.h5)
(available [here](https://github.ubc.ca/mds-2021-22/datasets/blob/master/data/food.zip))
- [Credit Card Dataset for Clustering](https://www.kaggle.com/arjunbhasin2013/ccdata)
- [Countries of the World](https://www.kaggle.com/fernandol/countries-of-the-world)
- [Airline Sentiment](https://www.kaggle.com/jaskarancr/airline-sentiment-dataset)
- [Jester 1.7M jokes ratings dataset](https://www.kaggle.com/vikashrajluhaniwal/jester-17m-jokes-ratings-dataset)
- [Amazon ratings data](http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/ratings_Patio_Lawn_and_Garden.csv)

If you want to be extra prepared, you may want to download these datasets in advance and save them under the `lectures/data` directory in your local copy of the repository. 

## Labs 
During labs, you will be given time to work on your own or in groups. There will be a lot of opportunity for discussion and getting help during lab sessions. 

## Installation
 
We are providing you with a `conda` environment file which is available [here](env-dsci-563.yml). You can download this file and create a conda environment for the course and activate it as follows. 

```
conda env create -f env-dsci-563.yml
conda activate 563
```

In order to use this environment in `Jupyter`, you will have to install `nb_conda_kernels` in the environment where you have installed `Jupyter` (typically the `base` environment). You will then be able to select this new environment in `Jupyter`. If you're unable to see the environment in Jupyter, you might have to install the kernel manually. See the documentation [here](https://ipython.readthedocs.io/en/stable/install/kernel_install.html). For more details on this, refer to your [521 lecture 7](https://pages.github.ubc.ca/MDS-2023-24/DSCI_521_platforms-dsci_students/lectures/7-virtual-environments.html#).

I've only attempted to install this environment file on a few machines, and you may encounter issues with certain packages from the `yaml` file when executing the commands above. This is not uncommon and may suggest that the specified package version is not yet available for your operating system via `conda`. When this occurs, you have a couple of options:

1. Modify the local version of the `yaml` file to remove the line containing that package.
2. Create the environment without that package. 
3. Activate the environment and install the package manually either with `conda install` or `pip install` in the environment.   

_Note that this is not a complete list of the packages we'll be using in the course and there might be a few packages you will be installing using `conda install` later in the course. But this is a good enough list to get you started._ 


## Course communication
<details>
  <summary>Click to expand!</summary>

We all are here to help you learn and succeed in the course and the program. Here is how we'll be communicating with each other during the course. 

### Clarifications on the lecture notes or lab questions

If there is any clarification on the lecture material or lab questions, I'll post a message on our course channel and tag you. **It is your responsibility to read the messages whenever you are tagged.** (I know that there are too many things for you to keep track of. You do not have to read all the messages but please make sure to carefully read the messages whenever you are tagged.) 

### Questions on lecture material or labs

If you have questions about the lecture material or lab questions please post them on the course Slack channel rather than direct messaging me or the TAs. Here are the advantages of doing so: 
- You'll get a quicker response. 
- Your classmates will benefit from the discussion. 

When you ask your question on the course channel, please avoid tagging the instructor unless it's specific for the instructor (e.g., if you notice some mistake in the lecture notes). If you tag a specific person, other teaching team members or your colleagues are discouraged to respond. This will decrease the response rate on the channel. 

Please use some consistent convention when you ask questions on Slack to facilitate easy search for others or future you. For example, if you want to ask a question on Exercise 3.2 from Lab 1, start your post with the label `lab1-ex2.3`. Or if you have a question on lecture 2 material, start your post with the label `lecture2`. Once the question is answered/solved, you can add "(solved)" tag before the label (e.g., (solved) `lab1-ex2.3`). Do not delete your post even if you figure out the answer on your own. The question and the discussion can still be beneficial to others.  

### Questions related to grading

For each deliverable, after I return grades, I'll let you know who has graded what in our course Slack by opening an issue in the course GitHub repository. If you have questions related to grading
- First, make sure your concerns are reasonable (read the ["Reasonable grading concerns" policy](https://ubc-mds.github.io/policies/)). 
- If you believe that your request is reasonable, open a regrade request on Gradescope. 
- If you are unable to resolve the issue with the TA, send a Slack message to the instructor, including the appropriate TA in the conversation. 

### Questions related to your personal situation or talking about sensitive information
 
I am open for a conversation with you. If you want to talk about anything sensitive, please direct message me on Slack (and tag me) rather than posting it on the course channel. It might take a while for me to get back to you, but I'll try my best to respond as soon as possible. 

</details>


## Reference Material
<details>
    <summary>Click to expand!</summary>   

### Books
* [A Course in Machine Learning (CIML)](http://ciml.info/) by Hal Daumé III (also relevant for DSCI 572, 573, 575, 563)
* Introduction to Machine Learning with Python: A Guide for Data Scientists by Andreas C. Mueller and Sarah Guido.
* [The Elements of Statistical Learning (ESL)](https://web.stanford.edu/~hastie/Papers/ESLII.pdf)
* [ML:APP](http://www.cs.ubc.ca/~murphyk/MLbook/index.html), 
* [LFD](http://amlbook.com/), 
* [AI:AMA](http://aima.cs.berkeley.edu/)
* [An Introduction to Statistical Learning](http://www-bcf.usc.edu/~gareth/ISL/ISLR%20Sixth%20Printing.pdf)

### Linear algebra review

- There are a bunch of suggestions [here](https://ubc-mds.github.io/resources_pages/learning_resources/). We particularly recommend [essence of linear algebra](https://www.youtube.com/watch?v=kjBOesZCoqc&list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab) (YouTube series) and
[Immersive linear algebra](http://immersivemath.com/ila/index.html) (interactive e-book).
- [Introduction to Linear Algebra for Applied Machine Learning with Python](https://pabloinsente.github.io/intro-linear-algebra)

### Online courses

* [Mike's CPSC 340](https://ubc-cs.github.io/cpsc340/)
* [Machine Learning](https://www.coursera.org/learn/machine-learning) (Andrew Ng's famous Coursera course)
* [Foundations of Machine Learning](https://bloomberg.github.io/foml/#home) online course from Bloomberg.
* [Machine Learning Exercises In Python, Part 1](http://www.johnwittenauer.net/machine-learning-exercises-in-python-part-1/) (translation of Andrew Ng's course to Python, also relevant for DSCI 561, 572, 563)

</details> 
  
## Policies

Please see the general [MDS policies](https://ubc-mds.github.io/policies/).
