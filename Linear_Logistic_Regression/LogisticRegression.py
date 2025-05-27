import numpy as np
from sklearn.preprocessing import OneHotEncoder

class LogisticRegression:
    def __init__(self, batch_size=32, regularization=0, max_epochs=100, patience=3):
        
        """Logistic Regression

        Parameters:
        -----------
        batch_size: int
            The number of samples per batch.
        regularization: float
            The regularization parameter.
        max_epochs: int
            The maximum number of epochs.
        patience: int
            The number of epochs to wait before stopping if the validation loss
            does not improve.
        """
        
        self.batch_size = batch_size
        self.regularization = regularization
        self.max_epochs = max_epochs
        self.patience = patience
        self.learning_rate = 0.01 # alpha
        self.weights = None
        self.bias = None
        self.loss_history = []

    def softmax(self, z):
        z_exp = np.exp(z-np.max(z, axis=1, keepdims=True))
        return z_exp/np.sum(z_exp,axis=1, keepdims=True)
        
        
    def crossEntropyloss(self, y_true, y_predicted):
        return -np.mean(np.sum(y_true * np.log(y_predicted),axis=1))      


    def fit(self, X, y, batch_size=32, regularization=0, max_epochs=100, patience=3):
        """Fit a linear model.

        Parameters:
        -----------
        batch_size: int
            The number of samples per batch.
        regularization: float
            The regularization parameter.
        max_epochs: int
            The maximum number of epochs.
        patience: int
            The number of epochs to wait before stopping if the validation loss
            does not improve.
        """
        self.batch_size = batch_size
        self.regularization = regularization
        self.max_epochs = max_epochs
        self.patience = patience
    
            
        encoder = OneHotEncoder(sparse_output=False)
            
        num_samples, num_features = X.shape
        num_classes = len(np.unique(y))
        
        y_one_hot = encoder.fit_transform(y.reshape(-1,1))
        

        # TODO: Initialize the weights and bias based on the shape of X and y.
        if self.weights is None:
            limit = np.sqrt(6 / (num_features + num_classes))
            self.weights = np.random.uniform(-limit, limit, (num_features, num_classes))

        if self.bias is None:
            self.bias = np.zeros((1,num_classes))
        
        val_split_idx = int(0.9* num_samples)
        
        x_train, x_val = X[:val_split_idx], X[val_split_idx:]
        y_train, y_val = y_one_hot[:val_split_idx], y_one_hot[val_split_idx:]
        
        
        
        init_loss = float('inf')
        final_weights = self.weights.copy()
        final_bias = self.bias.copy()
        p_count = 0
        
        
        for epoch in range(self.max_epochs):
            rand_idxs = np.random.permutation(len(x_train))
            x_train = x_train[rand_idxs]
            y_train = y_train[rand_idxs]
            
            for i in range(0, len(x_train), self.batch_size):
                
                # min-batch to avoid empty batch
                batch_x = x_train[i: min(i+self.batch_size, len(x_train))]
                batch_y = y_train[i:min(i+self.batch_size, len(y_train))]
                
                m = len(batch_x) 
                
                z = np.dot(batch_x, self.weights) + self.bias
                
                y_hat = self.softmax(z)
                
                error = y_hat - batch_y
                  
                grad_weights = (1/m) * np.dot(batch_x.T, error)
                grad_bias = (1/m) * np.sum(error, axis=0, keepdims=True)   
                
                if self.regularization > 0:    
                    grad_weights+= self.regularization * self.weights
                
                self.weights -= self.learning_rate * grad_weights
                self.bias -= self.learning_rate * grad_bias
                
            val_predict = np.dot(x_val, self.weights) + self.bias
            val_z = self.softmax(val_predict)
            val_loss = self.crossEntropyloss(y_val, val_z)     
            
            
            if val_loss < init_loss:
                init_loss = val_loss
                final_weights = self.weights.copy()
                final_bias = self.bias.copy()
                p_count = 0
            else:
                p_count+=1
                if p_count >= patience:
                    print('Early stopping....')
                    self.weights = final_weights
                    self.bias = final_bias
                    break    

    def predict(self, X):
        z = np.dot(X, self.weights) + self.bias 
        
        y_pred = self.softmax(z)
        
        return np.argmax(y_pred, axis=1)
    
    def score(self, X, y):
        y_pred = self.predict(X)
     
        return np.mean(y_pred == y)
       

    def save(self, file_path):
        #saves the model parameters to a file
        
        '''
            https://numpy.org/doc/stable/reference/generated/numpy.savez.html
        '''
        
        np.savez(file_path, weights = self.weights, bias = self.bias)
    
    def load(self,file_path):
        
        # loads model parameters from a file path
        '''
            https://numpy.org/doc/stable/reference/generated/numpy.load.html
        '''
        
        params = np.load(file_path)
        self.weights = params['weights']
        self.bias = params['bias']
        
        
        