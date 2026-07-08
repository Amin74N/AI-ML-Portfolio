import torch
def main():
    print("CPU tensor:")
    x = torch.randn(3,4)
    print(x)
    print("Type:", type(x), "\n--------------------------------------------------------------------------------")
    print("Converting CPU tensor to NumPy array:")
    y = x.numpy()
    print(y)
    print("Type:", type(y), "\n--------------------------------------------------------------------------------")
    print("Converting NumPy array to CPU tensor:")
    x = torch.from_numpy(y)
    print(x)
    print("Type:", type(x))
if __name__ == "__main__":
    main()
