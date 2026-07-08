import torch
def main():
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("CUDA is available.")
    else:
        device = torch.device("cpu")
        print("CUDA is not available.")
    x = torch.randn(2,3,5, device= device)
    print("Device:", x.device)
    print(x)
if __name__ == "__main__":
    main()
