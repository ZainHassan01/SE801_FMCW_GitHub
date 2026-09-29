import numpy as np

DATASET = r"C:\Users\Eng Zain\Desktop\Spring 26\ann\data_SAAB_SIRS_77GHz_FMCW.npy"

data = np.load(DATASET, allow_pickle=True)

print("Dataset:", data.shape, data.dtype)

for measurement_id in range(3):
    print(f"\nMeasurement {measurement_id}")

    for column in range(6):
        value = data[measurement_id, column]
        array = np.asarray(value)

        print(
            f"Column {column}:",
            f"type={type(value)}",
            f"shape={array.shape}",
            f"dtype={array.dtype}",
        )