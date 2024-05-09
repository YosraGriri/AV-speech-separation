import numpy as np
from pypesq import pesq
print('Hello')
# Create dummy audio signals
ref = np.random.randn(16000).astype(np.float32)
deg = np.random.randn(16000).astype(np.float32)
print(ref.shape, deg.shape)
pesq_score = 0
try:
    pesq_score = pesq(deg, ref, 16000)
except Exception as e:
    print(f"Error during PESQ calculation: {e}")
# Calculate PESQ score

print("PESQ score:", pesq_score)
