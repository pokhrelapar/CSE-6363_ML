
import numpy as np


class LinearRegression:
    def __init__(self, batch_size=32, regularization=0, max_epochs=100, patience=3):
        """Linear Regression using Gradient Descent.

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
        self.learning_rate = 0.01 #alpha
        self.weights = None
        self.bias = None
        self.loss_history = []

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
        
      
        y = y.reshape(-1,1) if y.ndim == 1 else y
        
        # print(y.shape)
        # print(X.shape)
            
        num_samples, num_features = X.shape
        num_outs = y.shape[1]

        #Xavier initilization
        if self.weights is None:
            limit = np.sqrt(6 / (num_features + num_outs))
            self.weights = np.random.uniform(-limit, limit, (num_features, num_outs))
        
        if self.bias is None:
            self.bias = np.zeros((1,num_outs))
        
        val_split_idx = int(0.9* num_samples)
        
        x_train, x_val = X[:val_split_idx], X[val_split_idx:]
        y_train, y_val = y[:val_split_idx].reshape(-1,1), y[val_split_idx:]
        
        
        init_loss = float('inf')
        final_weights = self.weights.copy()
        final_bias = self.bias.copy()
        p_count = 0
        
        
        for epoch in range(self.max_epochs):
            rand_idxs = np.random.permutation(len(x_train))
            x_train = x_train[rand_idxs]
            y_train = y_train[rand_idxs]
            
            for i in range(0, len(x_train), self.batch_size):
                
                batch_x = x_train[i:i+self.batch_size]
                batch_y = y_train[i:i+self.batch_size].reshape(-1,1)
                m = len(batch_x)
                
                y_hat = np.dot(batch_x,self.weights) + self.bias
                
                error = y_hat - batch_y
                
                # Mean squared error w/o regularization
                mse_loss = np.mean((y_hat - batch_y) ** 2)
                
                # Mean squared error with regularization
                total_loss = mse_loss + self.regularization * np.sum(self.weights**2)
                
                self.loss_history.append(total_loss) 
               
                       
                grad_weights = (1/m) * np.dot(batch_x.T, error)
                grad_bias = (1/m) * np.sum(error)   
                
                if self.regularization > 0:    
                    grad_weights+= 2 * self.regularization * self.weights
                
                #gradient update
                self.weights -= self.learning_rate * grad_weights
                self.bias -= self.learning_rate * grad_bias
                
            val_predict = np.dot(x_val, self.weights) + self.bias
            val_loss = np.mean((val_predict - y_val)**2) + self.regularization * np.sum(self.weights**2)         
            
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
        """Predict using the linear model.

        Parameters
        ----------
        X: numpy.ndarray
            The input data.
        """
        # TODO: Implement the prediction function.
        
        # y = xW + b
        return np.dot(X, self.weights) + self.bias
    
        

    def score(self, X, y):
        """Evaluate the linear model using the mean squared error.

        Parameters
        ----------
        X: numpy.ndarray
            The input data.
        y: numpy.ndarray
            The target data.
        """
        # TODO: Implement the scoring function.
        
        y = y.reshape(-1,1) if y.ndim == 1 else y
        
        y_hat = self.predict(X)
        mse_error = np.mean((y_hat - y)**2)
        return mse_error
    
    
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
        
        
        