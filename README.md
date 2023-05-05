Download Link: https://assignmentchef.com/product/solved-ece219-project-4-regression-analysis
<br>
Regression analysis is a statistical procedure for estimating the relationship between a target variable and a set of describing features. In this project, we explore common practices for best performance of regression. You will conduct different experiments and identify the significance of practices that we suggest below.

As for your report, whenever you encounter several options, explain what choices are better to make. You need to justify your claim and back it with numerical results. You are not asked to experiment all combinations of different choices in a tedious way. Instead, in some cases you may just report how an observations lead you to make a particular choice and move forward with that choice fixed. Casually speaking, you may at times make greedy decisions over the tree of your choices!

Answer all questions very concisely and to-the-point. Most questions require short answers with few keywords.

<h2>2       Datasets</h2>

You should take steps in section 3 on the following datasets.

<h3>2.1      Bike Sharing Dataset</h3>

This dataset can be downloaded from this <a href="https://archive.ics.uci.edu/ml/datasets/Bike+Sharing+Dataset"><strong>link</strong></a><a href="https://archive.ics.uci.edu/ml/datasets/Bike+Sharing+Dataset">.</a> Bike sharing dataset provides count number of rental bikes based on some timing and environmental conditions. You can find feature descriptions in the provided link.

We are asking to do all your analysis based on three different labels which are:

<ul>

 <li>casual: count of casual users</li>

 <li>registered: count of registered users</li>

 <li>cnt: count of total rental bikes including both casual and registered</li>

</ul>

Therefore, you have three targets. Do 3.1.1 data inspection for all three targets and continue project with total count (third item). As you might notice, there are two set of data, hour.csv and day.csv. Use day.csv for this project.

<h3>2.2       Video Transcoding Time Dataset</h3>

Video transcoding time dataset can be downloaded from this <a href="https://archive.ics.uci.edu/ml/machine-learning-databases/00335/"><strong>link</strong></a><a href="https://archive.ics.uci.edu/ml/machine-learning-databases/00335/">.</a> It includes input and output video characteristics along with their time taken for different valid transcodings. The dataset has 68784 data points in total, and each instance has 19 features as described below:

<ul>

 <li>duration = duration of video in second;</li>

 <li>codec = coding standard used for the input video (e.g. flv, h264, mpeg4, vp8);</li>

 <li>height, width = height and width of video in pixels;</li>

 <li>bitrate = bits that are conveyed or processed per unit of time for input video;</li>

 <li>frame rate = input video frame rate(fps);</li>

 <li>i = number of i frames in the video, where i frames are the least compressible but don’t require other video frames to decode;</li>

 <li>p = number of p frames in the video, where p frames can use data from previous frames to decompress and are more compressible than I-frames;</li>

 <li>b = number of b frames in the video, where b frames can use both previous and forward frames for data reference to get the highest amount of data compression;</li>

 <li>frames = number of frames in video;</li>

 <li>i-size, p-size, b-size = total size in byte of i, p, b frames;</li>

 <li>size = total size of video in byte;</li>

 <li>o-codec = output codec used for transcoding;</li>

 <li>o-bitrate = output bitrate used for transcoding;</li>

 <li>o-framerate = output framerate used for transcoding;</li>

 <li>o-width, o-height = output width and height in pixel used for transcoding.</li>

</ul>

There are two files in the downloaded folder. Only use transcoding-mesurment.tsv for your project. Please notice that this file contains 19 features above as well as following two attributes:

<ul>

 <li>umem = total codec allocated memory for transcoding;</li>

 <li>utime = total transcoding time for transcoding.</li>

</ul>

Note that the target variable is transcoding time, which is the last attribute “utime” in the data file.

<h2>3       Required Steps</h2>

In this section, we describe the setup you need to follow. Take these steps on the datasets in section 2. (Take whichever steps that may apply to each dataset.).

<h3>3.1      Before Training</h3>

Before training an algorithm, it’s always essential to inspect data and understand how it looks like. Also, raw data might need some preprocessing. In this section we will address these steps.

<h4>3.1.1        Data Inspection</h4>

The first step for data analysis is to take a close look at the dataset.

<ul>

 <li>Plot a heatmap of Pearson correlation matrix of dataset columns. Report which features have the highest absolute correlation with the target variable and what that implies. <strong>(Question 1)</strong></li>

 <li>Plot the histogram of numerical features. What preprocessing can be done if the distribution of a feature has high skewness? <strong>(Question 2)</strong></li>

 <li>Inspect box plot of categorical features vs target variable. What intuition do you get? <strong>(Question 3)</strong></li>

 <li>For bike sharing dataset, plot the count number per day for a few months. Can you identify any repeating patterns in every month? <strong>(Question 4)</strong></li>

 <li>For video transcoding time dataset, plot the distribution of video transcoding times, what can you observe? Report mean and median transcoding times. <strong>(Question 5)</strong></li>

</ul>

<h4>3.1.2         Handling Categorical Features</h4>

A categorical feature is a feature that can take on one of a limited number of possible values. A preprocessing step is to convert categorical variables into numbers and thus prepared for training.

One method for numerical encoding of categorical features is to assign a scalar. For instance, if we have a “Quality” feature with values {Poor, Fair, Typical, Good, Excellent} we might replace them with numbers 1 through 5. If there is no numerical meaning behind categorical features (e.g. {Cat, Dog}) one has to perform “one-hot encoding” instead.

For some other cases, e.g. when encoding time stamps such as {Mon, …, Sun} or {Jan, …, Dec} it might make sense to perform either one. In those cases the learning algorithm of choice and numerical results can lead our way. Can you explain a trade-off here? (Hint: let us assume we perform linear regression, what information does one-hot encoding discard, and what assumption should hold strongly if we perform the scalar encoding instead?) <strong>(Question 6)</strong>

<h4>3.1.3       Standardization</h4>

Standardization of datasets is a common requirement for many machine learning estimators; they might behave badly if the individual features do not more-or-less look like standard normally distributed data: Gaussian with zero mean and unit variance. If a feature has a variance that is orders of magnitude larger than others, it might dominate the objective function and make the estimator unable to learn from other features correctly as expected.

Standardize feature columns and prepare them for training. <strong>(Question 7)</strong>

<h4>3.1.4       Feature Selection</h4>

<ul>

 <li>feature selection.mutual info regression function returns estimated mutual information between each feature and the label. Mutual information (MI) between two random variables is a non-negative value which measures the dependency between the variables. It is equal to zero if and only if two random variables are independent, and higher values mean higher dependency.</li>

 <li>feature selection.f regression function provides F scores, which is a way of comparing the significance of the improvement of a model, with respect to the addition of new variables.</li>

</ul>

You may use these functions to select most important features. How does this step affect the performance of your models in terms of test RMSE? <strong>(Question 8)</strong>

<h3>3.2     Training</h3>

Once the data is prepared, we would like to train multiple algorithms and compare their performance using RMSE (please refer to part 3.3).

<h4>3.2.1        Linear Regression</h4>

What is the objective function? Train ordinary least squares (linear regression without regularization), as well as Lasso and Ridge regression, and compare their performances. Answer the following questions.

<ul>

 <li>Explain how each regularization scheme affects the learned hypotheses. <strong>(Question 9)</strong></li>

 <li>Report your choice of the best regularization scheme along with the optimal penalty parameter and briefly explain how it can be computed. <strong>(Question 10)</strong></li>

 <li>Does feature scaling play any role (in the cases with and without regularization)? Justify your answer. <strong>(Question 11)</strong></li>

 <li>Some linear regression packages return <em>p</em>-values for different features<a href="#_ftn1" name="_ftnref1"><sup>[1]</sup></a>. What is the meaning of them and how can you infer the most significant features? <strong>(Question 12)</strong></li>

</ul>

<h4>3.2.2        Polynomial Regression</h4>

Perform polynomial regression by crafting products of raw features up to a certain degree and applying linear regression on the compound features. You can use scikit-learn library to build such features. Avoid overfitting by proper regularization. Answer the following:

<ul>

 <li>Look up for the most salient features and interpret them. <strong>(Question 13)</strong></li>

 <li>What degree of polynomial is best? What reasons would stop us from too much increase of the polynomial degree? How do you choose that? <strong>(Question 14)</strong></li>

 <li>For the transcoding dataset it might make sense to craft inverse of certain features such that you get features such as, etc. Explain why this might make sense and check if doing so will boost accuracy. <strong>(Question 15)</strong></li>

</ul>

<h4>3.2.3        Neural Network</h4>

Try a multi-layer perceptron (fully connected neural network). You can simply use sklearn implementation and compare the performance. Then answer the following:

<ul>

 <li>Why does it do much better than linear regression? <strong>(Question 16)</strong></li>

 <li>Adjust your network size (number of hidden neurons and depth), and weight decay as regularization. Find a good hyper-parameter set systematically. <strong>(Question 17)</strong></li>

 <li>What activation function should be used for the output? You may use none. <strong>(Question 18)</strong></li>

 <li>What reasons would stop us from too much increase of the depth of the network? <strong>(Question 19)</strong></li>

</ul>

<h4>3.2.4        Random Forest</h4>

Apply a random forest regression model on datasets, and answer the following.

<ul>

 <li>Random forests have the following hyper-parameters:

  <ul>

   <li>Maximum number of features;</li>

   <li>Number of trees;</li>

   <li>Depth of each tree;</li>

  </ul></li>

</ul>

Fine-tune your model. Explain how these hyper-parameters affect the overall performance? Do some of them have regularization effect? <strong>(Question 20)</strong>

<ul>

 <li>Why does random forest perform well? <strong>(Question 21)</strong></li>

 <li>Randomly pick a tree in your random forest model (with maximum depth of 4) and plot its structure. Which feature is selected for branching at the root node? What can you infer about the importance of features? Do the important features match what you got in part 3.2.1? <strong>(Question 22)</strong></li>

</ul>

<h3>3.3     Evaluation</h3>

Perform 10-fold cross-validation and measure average RMSE errors for training and validation sets. Why is the training RMSE different from that of validation set? <strong>(Question 23)</strong>

For random forest model, measure “Out-of-Bag Error” (OOB) as well. Explain what OOB error and <em>R</em><sup>2 </sup>score means given this <a href="https://github.com/scikit-learn/scikit-learn/blob/b194674c4/sklearn/ensemble/_forest.py#L788"><strong>link</strong></a><a href="https://github.com/scikit-learn/scikit-learn/blob/b194674c4/sklearn/ensemble/_forest.py#L788">.</a> <strong>(Question 24)</strong>

<a href="#_ftnref1" name="_ftn1">[1]</a> E.g: scipy.stats.linregress and statsmodels.regression.linear model.OLS