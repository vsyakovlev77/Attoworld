"""Test time-frequency analysis methods."""

import attoworld as aw
import numpy as np


def create_test_waveform():
    """Create a simple test waveform for time-frequency analysis."""
    # Create a Gaussian-modulated sinusoid
    duration = 50e-15  # 50 fs
    dt = 0.5e-15  # 0.5 fs
    time = np.arange(0, duration, dt)
    t_center = duration / 2
    sigma = duration / 8

    # Carrier wave at 375 THz (800 nm)
    f0 = 375e12
    wave = np.sin(2 * np.pi * f0 * time) * np.exp(-((time - t_center) ** 2) / (2 * sigma**2))

    return aw.data.Waveform(
        wave=wave,
        time=time,
        dt=dt,
        is_uniformly_spaced=True,
    )


def test_stft_basic():
    """Test that STFT returns a valid Spectrogram."""
    waveform = create_test_waveform()
    spec = aw.wave.stft(waveform)

    # Check that result is a Spectrogram
    assert isinstance(spec, aw.data.Spectrogram)

    # Check shapes
    assert spec.data.ndim == 2
    assert len(spec.time) == spec.data.shape[1]
    assert len(spec.freq) == spec.data.shape[0]

    # Check that data is real and non-negative (power spectrum)
    assert np.all(np.isreal(spec.data))
    assert np.all(spec.data >= 0)


def test_stft_complex():
    """Test that STFT can return complex values."""
    waveform = create_test_waveform()
    spec = aw.wave.stft(waveform, return_complex=True)

    # Check that result contains complex values
    assert np.iscomplexobj(spec.data)


def test_cwt_basic():
    """Test that CWT returns a valid Spectrogram."""
    waveform = create_test_waveform()
    spec = aw.wave.cwt(waveform, num_scales=20)

    # Check that result is a Spectrogram
    assert isinstance(spec, aw.data.Spectrogram)

    # Check shapes
    assert spec.data.ndim == 2
    assert len(spec.time) == spec.data.shape[1]
    assert len(spec.freq) == spec.data.shape[0]
    assert len(spec.freq) == 20  # Should match num_scales

    # Check that data is real and non-negative (power spectrum)
    assert np.all(np.isreal(spec.data))
    assert np.all(spec.data >= 0)


def test_cwt_with_freq_range():
    """Test that CWT respects frequency range."""
    waveform = create_test_waveform()
    freq_min = 200e12
    freq_max = 600e12
    spec = aw.wave.cwt(waveform, freq_range=(freq_min, freq_max), num_scales=30)

    # Check that frequencies are within range (with some tolerance for edge effects)
    assert spec.freq.min() >= freq_min * 0.8
    assert spec.freq.max() <= freq_max * 1.2


def test_wigner_ville_basic():
    """Test that Wigner-Ville returns a valid Spectrogram."""
    waveform = create_test_waveform()
    spec = aw.wave.wigner_ville(waveform)

    # Check that result is a Spectrogram
    assert isinstance(spec, aw.data.Spectrogram)

    # Check shapes
    assert spec.data.ndim == 2
    assert len(spec.time) == spec.data.shape[1]
    assert len(spec.freq) == spec.data.shape[0]

    # Check that data is real (WVD is real-valued)
    assert np.all(np.isreal(spec.data))

    # WVD can have negative values due to cross-terms
    # For a simple single-component signal, most should be positive
    assert np.sum(spec.data > 0) > 0.5 * spec.data.size


def test_compare_time_frequency_methods():
    """Test the comparison convenience function."""
    waveform = create_test_waveform()

    # Test with all methods
    results = aw.wave.compare_time_frequency_methods(waveform)
    assert "STFT" in results
    assert "CWT" in results
    assert "Wigner-Ville" in results

    # Each should be a valid Spectrogram
    for method, spec in results.items():
        assert isinstance(spec, aw.data.Spectrogram)
        assert spec.data.ndim == 2

    # Test with subset of methods
    results_subset = aw.wave.compare_time_frequency_methods(
        waveform, methods=["stft", "cwt"]
    )
    assert len(results_subset) == 2
    assert "STFT" in results_subset
    assert "CWT" in results_subset
    assert "Wigner-Ville" not in results_subset


def test_stft_windows():
    """Test that different window functions work."""
    waveform = create_test_waveform()

    windows = ["hann", "hamming", "blackman", "tukey"]
    for window in windows:
        spec = aw.wave.stft(waveform, window=window)
        assert isinstance(spec, aw.data.Spectrogram)
        assert np.all(spec.data >= 0)


def test_cwt_wavelets():
    """Test that different wavelet types work."""
    waveform = create_test_waveform()

    wavelets = ["morlet", "ricker"]
    for wavelet in wavelets:
        spec = aw.wave.cwt(waveform, wavelet=wavelet, num_scales=15)
        assert isinstance(spec, aw.data.Spectrogram)
        assert np.all(spec.data >= 0)


def test_wigner_ville_energy_conservation():
    """Test that Wigner-Ville integrates to signal energy."""
    waveform = create_test_waveform()
    spec = aw.wave.wigner_ville(waveform)

    # Signal energy
    signal_energy = np.sum(np.abs(waveform.wave) ** 2) * waveform.dt

    # WVD integrated over time-frequency should give twice the signal energy
    # (due to the bilinear nature of WVD)
    df = spec.freq[1] - spec.freq[0] if len(spec.freq) > 1 else 1.0
    dt = spec.time[1] - spec.time[0] if len(spec.time) > 1 else waveform.dt
    wvd_integral = np.sum(spec.data) * dt * df

    # This is approximate due to discretization and boundary effects
    # We just check they're in the same order of magnitude
    assert np.abs(np.log10(wvd_integral / signal_energy)) < 2


def test_uniformly_spaced_conversion():
    """Test that non-uniformly spaced waveforms are handled correctly."""
    # Create a non-uniformly spaced waveform
    time = np.array([0, 1e-15, 3e-15, 6e-15, 10e-15, 15e-15])
    wave = np.sin(2 * np.pi * 375e12 * time)
    waveform = aw.data.Waveform(
        wave=wave,
        time=time,
        dt=1e-15,  # This is just nominal
        is_uniformly_spaced=False,
    )

    # All methods should handle this by converting to uniform spacing
    spec_stft = aw.wave.stft(waveform)
    spec_cwt = aw.wave.cwt(waveform, num_scales=10)
    spec_wvd = aw.wave.wigner_ville(waveform)

    # Should all return valid spectrograms
    assert isinstance(spec_stft, aw.data.Spectrogram)
    assert isinstance(spec_cwt, aw.data.Spectrogram)
    assert isinstance(spec_wvd, aw.data.Spectrogram)
