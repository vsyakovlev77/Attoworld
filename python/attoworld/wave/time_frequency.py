"""Time-frequency analysis methods for waveforms.

This module provides various time-frequency analysis techniques including:
- Short-Time Fourier Transform (STFT)
- Continuous Wavelet Transform (CWT)
- Wigner-Ville Distribution (WVD)

All methods return Spectrogram objects with time and frequency axes in SI units.
"""

from typing import Optional, Tuple

import numpy as np
import scipy.signal as sig

from ..data import Spectrogram, Waveform


def stft(
    waveform: Waveform,
    window: str = "hann",
    nperseg: Optional[int] = None,
    noverlap: Optional[int] = None,
    nfft: Optional[int] = None,
    return_complex: bool = False,
) -> Spectrogram:
    """Compute the Short-Time Fourier Transform of a waveform.

    The STFT divides the signal into short overlapping segments and computes
    the Fourier transform of each segment, providing time-frequency localization.

    Args:
        waveform: Input Waveform object
        window: Window function to use. Can be any string accepted by scipy.signal.get_window
            (e.g., 'hann', 'hamming', 'blackman', 'tukey'). Default is 'hann'.
        nperseg: Length of each segment. If None, uses 256 samples or length//8,
            whichever is smaller.
        noverlap: Number of points to overlap between segments. If None, uses nperseg//2.
        nfft: Length of the FFT used. If None, uses nperseg.
        return_complex: If True, returns complex-valued STFT. If False (default),
            returns magnitude squared (power spectrogram).

    Returns:
        Spectrogram: Time-frequency representation with:
            - data: 2D array of STFT coefficients (power or complex)
            - time: Time vector (s)
            - freq: Frequency vector (Hz)

    Examples:
        >>> # Basic STFT with default settings
        >>> spec = stft(waveform)
        >>> spec.plot()
        >>>
        >>> # STFT with custom window and segment length
        >>> spec = stft(waveform, window='blackman', nperseg=512)
        >>>
        >>> # Complex-valued STFT for phase analysis
        >>> spec_complex = stft(waveform, return_complex=True)

    Notes:
        - The waveform is automatically converted to uniformly spaced if needed
        - Uses scipy.signal.stft internally
        - Time-frequency resolution tradeoff controlled by nperseg:
          larger nperseg = better frequency resolution, worse time resolution
    """
    # Convert to uniformly spaced waveform
    uniform_waveform = waveform.to_uniformly_spaced()

    # Compute STFT using scipy
    f, t, Zxx = sig.stft(
        uniform_waveform.wave,
        fs=1.0 / uniform_waveform.dt,
        window=window,
        nperseg=nperseg,
        noverlap=noverlap,
        nfft=nfft,
        return_onesided=False,
        boundary=None,
        padded=False,
    )

    # Shift frequency axis to centered and convert to absolute frequencies
    f_shifted = np.fft.fftshift(f)
    Zxx_shifted = np.fft.fftshift(Zxx, axes=0)

    # Adjust time axis to match original waveform time offset
    t_adjusted = t + uniform_waveform.time[0]

    # Prepare output data
    if return_complex:
        data = Zxx_shifted
    else:
        # Power spectrogram (magnitude squared)
        data = np.abs(Zxx_shifted) ** 2

    return Spectrogram(data=data, time=t_adjusted, freq=f_shifted)


def cwt(
    waveform: Waveform,
    wavelet: str = "morlet",
    scales: Optional[np.ndarray] = None,
    num_scales: int = 100,
    freq_range: Optional[Tuple[float, float]] = None,
) -> Spectrogram:
    """Compute the Continuous Wavelet Transform of a waveform.

    The CWT provides time-frequency analysis with variable time-frequency resolution:
    better time resolution at high frequencies, better frequency resolution at low frequencies.

    Args:
        waveform: Input Waveform object
        wavelet: Wavelet type. Options:
            - 'morlet': Morlet wavelet (default, good for most purposes)
            - 'ricker': Ricker wavelet (Mexican hat)
            - 'morlet2': Complex Morlet wavelet with bandwidth parameter
        scales: Array of scales to use. If None, automatically generated from freq_range
            or num_scales.
        num_scales: Number of scales to use if scales not provided. Default is 100.
        freq_range: Tuple of (min_freq, max_freq) in Hz. If None, uses full range
            based on sampling rate.

    Returns:
        Spectrogram: Time-frequency representation with:
            - data: 2D array of CWT coefficients (power)
            - time: Time vector (s), same as input waveform
            - freq: Frequency vector (Hz) corresponding to scales

    Examples:
        >>> # Basic CWT with Morlet wavelet
        >>> spec = cwt(waveform)
        >>> spec.plot()
        >>>
        >>> # CWT with specific frequency range
        >>> spec = cwt(waveform, freq_range=(100e12, 1000e12))
        >>>
        >>> # CWT with Ricker wavelet
        >>> spec = cwt(waveform, wavelet='ricker', num_scales=128)

    Notes:
        - The waveform is automatically converted to uniformly spaced if needed
        - Uses scipy.signal.cwt internally
        - For Morlet wavelet, central frequency is assumed to be ω₀ = 6 rad/s
        - Scale-to-frequency conversion: f = f_center / (scale * dt)
    """
    # Convert to uniformly spaced waveform
    uniform_waveform = waveform.to_uniformly_spaced()

    dt = uniform_waveform.dt
    fs = 1.0 / dt  # Sampling frequency
    N = len(uniform_waveform.wave)

    # Determine wavelet center frequency for scale-to-frequency conversion
    if wavelet == "morlet" or wavelet == "morl":
        # Morlet wavelet with ω₀ = 6
        wavelet_center_freq = 6.0 / (2 * np.pi)  # In normalized frequency
        wavelet_func = sig.morlet2
        wavelet_kwargs = {"w": 6.0}
    elif wavelet == "ricker":
        # Ricker wavelet (2nd derivative of Gaussian)
        wavelet_center_freq = 1.0 / np.sqrt(2 * np.pi)  # Approximate
        wavelet_func = sig.ricker
        wavelet_kwargs = {}
    else:
        raise ValueError(
            f"Unknown wavelet '{wavelet}'. Supported: 'morlet', 'ricker', 'morlet2'"
        )

    # Determine scales
    if scales is None:
        if freq_range is not None:
            # Convert frequency range to scale range
            # f = wavelet_center_freq * fs / scale
            # scale = wavelet_center_freq * fs / f
            scale_min = wavelet_center_freq * fs / freq_range[1]
            scale_max = wavelet_center_freq * fs / freq_range[0]
        else:
            # Use reasonable default: from Nyquist/2 to 1/8 of sampling rate
            scale_min = 2
            scale_max = N / 4

        # Generate logarithmically spaced scales
        scales = np.logspace(np.log10(scale_min), np.log10(scale_max), num_scales)

    # Compute CWT
    if wavelet == "ricker":
        # Ricker uses different signature - points parameter for scale
        cwt_matrix = np.zeros((len(scales), N), dtype=complex)
        for i, scale in enumerate(scales):
            # Generate Ricker wavelet at this scale
            points = min(int(10 * scale), N)
            if points % 2 == 0:
                points += 1  # Make odd
            wavelet_data = wavelet_func(points, scale)
            # Convolve with signal
            cwt_matrix[i, :] = sig.fftconvolve(
                uniform_waveform.wave, wavelet_data, mode="same"
            )
    else:
        # Morlet wavelets - use scipy's cwt
        # Note: scipy.signal.cwt expects widths parameter which is like scales
        cwt_matrix = sig.cwt(uniform_waveform.wave, wavelet_func, scales, **wavelet_kwargs)

    # Convert scales to frequencies
    freqs = wavelet_center_freq * fs / scales

    # Sort frequencies in ascending order (scales are descending)
    sort_idx = np.argsort(freqs)
    freqs = freqs[sort_idx]
    cwt_matrix = cwt_matrix[sort_idx, :]

    # Compute power
    data = np.abs(cwt_matrix) ** 2

    return Spectrogram(
        data=data, time=uniform_waveform.time, freq=freqs
    )


def wigner_ville(
    waveform: Waveform,
    nfft: Optional[int] = None,
) -> Spectrogram:
    """Compute the Wigner-Ville Distribution of a waveform.

    The Wigner-Ville Distribution (WVD) is a time-frequency representation that
    provides optimal time-frequency localization but suffers from cross-terms
    when analyzing multi-component signals.

    The WVD is defined as:
    W(t, f) = ∫ x(t + τ/2) x*(t - τ/2) e^(-j2πfτ) dτ

    Args:
        waveform: Input Waveform object
        nfft: Length of FFT to use. If None, uses 2 * len(waveform) - 1 for
            better frequency resolution.

    Returns:
        Spectrogram: Time-frequency representation with:
            - data: 2D array of WVD (real-valued, can have negative values due to interference)
            - time: Time vector (s), same as input waveform
            - freq: Frequency vector (Hz)

    Examples:
        >>> # Basic Wigner-Ville distribution
        >>> spec = wigner_ville(waveform)
        >>> spec.plot()
        >>>
        >>> # WVD with custom FFT length
        >>> spec = wigner_ville(waveform, nfft=1024)

    Notes:
        - The waveform is automatically converted to uniformly spaced if needed
        - WVD is real-valued but can be negative due to cross-term interference
        - For multi-component signals, consider using smoothed pseudo-Wigner-Ville
          or other reduced-interference distributions
        - The WVD is bilinear and satisfies many desirable mathematical properties
        - Cross-terms appear at the geometric mean of component frequencies

    References:
        - Cohen, L. (1995). Time-frequency analysis (Vol. 778). Prentice hall.
        - Flandrin, P. (1999). Time-frequency/time-scale analysis. Academic press.
    """
    # Convert to uniformly spaced waveform
    uniform_waveform = waveform.to_uniformly_spaced()

    signal = uniform_waveform.wave
    N = len(signal)
    dt = uniform_waveform.dt
    fs = 1.0 / dt

    # Determine FFT length
    if nfft is None:
        nfft = 2 * N - 1

    # Initialize WVD matrix
    wvd = np.zeros((nfft, N), dtype=complex)

    # Compute WVD for each time point
    for n in range(N):
        # Determine valid range for lag parameter τ
        # We need both n+m and n-m to be valid indices
        tau_max = min(n, N - 1 - n)

        # Create the instantaneous autocorrelation function
        # R[n,m] = x[n+m] * conj(x[n-m])
        R = np.zeros(nfft, dtype=complex)

        # Handle zero lag (m=0)
        R[0] = signal[n] * np.conj(signal[n])

        # Compute for positive and negative lags symmetrically
        for m in range(1, tau_max + 1):
            # Positive lag
            R[m] = signal[n + m] * np.conj(signal[n - m])
            # Negative lag (use conjugate symmetry)
            R[nfft - m] = np.conj(R[m])

        # Take FFT to get frequency representation
        wvd[:, n] = np.fft.fft(R)

    # Take real part (imaginary part should be ~0 due to symmetry)
    # and shift to center zero frequency
    wvd_real = np.real(wvd)
    wvd_shifted = np.fft.fftshift(wvd_real, axes=0)

    # Create frequency axis
    freqs = np.fft.fftshift(np.fft.fftfreq(nfft, d=dt))

    return Spectrogram(
        data=wvd_shifted,
        time=uniform_waveform.time,
        freq=freqs,
    )


def compare_time_frequency_methods(
    waveform: Waveform,
    methods: Optional[list] = None,
) -> dict:
    """Compare multiple time-frequency analysis methods on the same waveform.

    This convenience function applies multiple time-frequency methods and returns
    them in a dictionary for easy comparison and visualization.

    Args:
        waveform: Input Waveform object
        methods: List of methods to apply. Options: 'stft', 'cwt', 'wigner'.
            If None, applies all three methods with default parameters.

    Returns:
        dict: Dictionary with method names as keys and Spectrogram objects as values.
            Keys are: 'STFT', 'CWT', 'Wigner-Ville'

    Example:
        >>> results = compare_time_frequency_methods(waveform)
        >>> results['STFT'].plot()
        >>> results['CWT'].plot()
        >>> results['Wigner-Ville'].plot()
        >>>
        >>> # Compare only STFT and CWT
        >>> results = compare_time_frequency_methods(waveform, methods=['stft', 'cwt'])
    """
    if methods is None:
        methods = ["stft", "cwt", "wigner"]

    results = {}

    if "stft" in methods:
        results["STFT"] = stft(waveform)

    if "cwt" in methods:
        results["CWT"] = cwt(waveform)

    if "wigner" in methods:
        results["Wigner-Ville"] = wigner_ville(waveform)

    return results
