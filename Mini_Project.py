
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Dense, BatchNormalization, Activation
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

np.random.seed(42)

# 20×2000 integers in [-6, 6]
data = np.random.randint(-6, 7, size=(20, 2000))

# define 20 asymmetric functions
funcs = [
    lambda a, b: a - b,
    lambda a, b: a**2 - b,
    lambda a, b: 2*a - 3*b,
    lambda a, b: a*b - b,
    lambda a, b: a**3 - 2*b,
    lambda a, b: 5*a - b**2,
    lambda a, b: a**2 - 3*b,
    lambda a, b: 4*a - b**3,
    lambda a, b: a*b + a - b,
    lambda a, b: a**2 + b - a,
    lambda a, b: 3*a - 2*b**2,
    lambda a, b: a*3 - b*2,
    lambda a, b: 7*a - 5*b,
    lambda a, b: a*b**2 - b,
    lambda a, b: 2*a**2 - 3*b,
    lambda a, b: a*2 - b*4,
    lambda a, b: 9*a - 2*b,
    lambda a, b: a*b - a**2,
    lambda a, b: a**2 + 2*a - b,
    lambda a, b: 2*a**3 - b,
]

transformed = np.empty((20, 1000), dtype=np.int64)

for i, f in enumerate(funcs):
    # take pairs of columns (0,1), (2,3), ..., (1998,1999)
    for j in range(1000):
        a = data[i, 2*j]      # even column
        b = data[i, 2*j + 1]  # next odd column
        transformed[i, j] = f(a, b)

print(transformed.shape)      # check (20, 1000)
print(data[:2, :10])
print(transformed[:2, :10])    # internal check to see if funcs applied correctly

X = transformed.T.astype("float32")          # shape (1000, 20)

# standardize for each dimension, to keep the nature of the relationship the same, but with consistent parameters (mean = 0, std = 1)
mu  = X.mean(axis=0, keepdims=True)
std = X.std(axis=0, keepdims=True) + 1e-8
Xn  = (X - mu) / std #standardised X (Xnew)

# train/val split
idx = np.random.RandomState(42).permutation(len(Xn))
train_idx, val_idx = idx[:800], idx[800:]
Xtr, Xval = Xn[train_idx], Xn[val_idx]

# --- model ---
latent_dim = 8
inp = Input(shape=(20,))


# Define encoder
e = Dense(256, activation='relu')(inp) 
e = Dense(128, activation='relu')(e)
e = Dense(64, activation='relu')(e)  
z = Dense(latent_dim, activation='linear', name="latent")(e)   # 20 -> 2


# Define decoder
d = Dense(64, activation='relu')(z)
d = Dense(128, activation='relu')(d)
d = Dense(256, activation='relu')(d)
out = Dense(20, activation='linear')(d)                        # reconstruct 20D

auto = Model(inp, out)
auto.compile(optimizer=Adam(1e-4), loss='mse')

es = EarlyStopping(patience=20, restore_best_weights=True)
auto.fit(Xtr, Xtr, validation_data=(Xval, Xval),
         epochs=500, batch_size=32, callbacks=[es], shuffle=True)

# separate encoder/decoder if needed (since once AE is trained, encoder used to use BO on latent space, then decoder may be used reconstruct back to 20D)
encoder = Model(inp, z)

# decoder (once seperated and trained)

lat_layer = auto.get_layer('latent')
lat_idx   = auto.layers.index(lat_layer)

z_in = Input(shape=(latent_dim,))
x = z_in
for L in auto.layers[lat_idx + 1:]:   # all layers after 'latent' are decoder layers
    x = L(x)
decoder = Model(z_in, x)

# Z is the latent set we use for BO
Z = encoder.predict(Xn)                 # shape (1000, latent dim)

# reconstruct to 20D input from Xn (and unstandardize if you want original scale)
Xn_rec = auto.predict(Xn)
X_rec = Xn_rec * std + mu

# reconstruction error per sample (track this)
recon_mse = np.mean((Xn - Xn_rec)**2, axis=1)
print("Mean MSE:", recon_mse.mean())

hist = auto.history.history

# 1. loss vs epoch
plt.figure()
plt.plot(hist['loss'], label='train loss')
plt.plot(hist['val_loss'], label='val loss')
plt.ylim((0,1))
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.legend()
plt.title('Training vs Validation Loss')
plt.show()

