import numpy as np


random_vals = (np.random.rand(1001,2)-.5)*np.random.randint(low=0,high=10)
m_true,b_true = random_vals[0,:]
x_train,y_train = random_vals[1:801,:].transpose()
x_test, y_test = random_vals[802:,:].transpose()

x_bar = np.mean(x_train)
y_bar = np.mean(y_train)

m_pred = np.dot(x_bar-x_train,y_bar-y_train)/np.sum((x_train-x_bar)**2)
b_pred =  y_bar - x_bar * m_pred

y_pred = m_pred * x_test + b_pred
err = np.mean((y_test-y_pred)**2)


print('             True  |  Pred  ')
print(f'Slope:     {m_true:.4f} | {m_pred:.4f}')
print(f'Intercept: {b_true:.4f} | {b_pred:.4f}')
print('-'*30)
print(f'MSE: {err:.4f}')