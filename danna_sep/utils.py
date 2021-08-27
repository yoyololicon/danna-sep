import torch


def stft(x, n_fft=4096, n_hopsize=1024):
    window = torch.hann_window(n_fft, dtype=x.dtype, device=x.device)
    X = torch.stft(
        x,
        n_fft,
        n_hopsize,
        n_fft,
        window,
        return_complex=True
    )
    return X


def istft(X, n_fft=4096, n_hopsize=1024):
    dtype = X.dtype
    if dtype == torch.complex32:
        dtype = torch.float16
    elif dtype == torch.complex64:
        dtype = torch.float32
    elif dtype == torch.complex128:
        dtype = torch.float64
    window = torch.hann_window(n_fft, dtype=dtype, device=X.device)
    x = torch.istft(
        X,
        n_fft,
        n_hopsize,
        n_fft,
        window,
    )
    return x
