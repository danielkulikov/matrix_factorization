# matrix_factorization

homebrew implementations of matrix factorization for recommender systems, both for explicit feedback (trained with stochastic gradient descent) and implicit feedback (trained with alternating least squares). 

## Usage

```python
# create a matrix_factorizer object with the user-item rating matrix and some other arguments - choose either implicit or explicit feedback type. 
mf = matrix_factorizer(rating_matrix, params, dim, "implicit", i_matrix)
# trains the model 
mf.train()
# returns the user-feature and item-feature trained parameters. 
model = mf.get_model()
```

