# Linear Regression

1. What are the pros and cons of using the normal equation to solve for the weights in linear regression as opposed to using gradient descent?

   ## Pros: Normal Equation

   --> Normal equations skips the hassle of tuning the learning rate
   --> Normal equations are computationally cheaper for smaller datasets and gives exact solutions for weight
   --> They don't require iterative optimization and converge immediately

   ## Cons : Normal Equation

   --> Computing (X.TX)^-1 can be computationally expensive with a time complexity of O(d^3); d = # of features
   --> Unliked gradient descent which costs O(nd); n = # of samples, normal equations does not scale well with massive datasets

# Logistic Regression

1. Why is the softmax function used in multi-class logistic regression (Hint: the model itself produces logits)?

   Multi-class logistic regression model produces raw, unnormalized scores for each class. These can be any real number, +ve or -ve and rarely represent the probabilites directly.

   The softmax function covnerts these raw scores into probabilites. This as a result assigns a
   probability score to all the classes in such a way that the sum of all the probabilities is equal to 1. We can then intepret the outputs as the likelihood of each class.
