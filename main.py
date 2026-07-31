import sounddevice as sd
import numpy as np

SAMPLE_RATE = 44100
DURATION = 5

print("Recording...")

audio = sd.rec(
    int(DURATION * SAMPLE_RATE),
    samplerate=SAMPLE_RATE,
    channels=1,
    dtype="float32"
)

sd.wait()

audio = audio.flatten()

print("Recording finished!")

average_amplitude = np.mean(np.abs(audio))
print(f"Average amplitude: {average_amplitude:.4f}")

fft = np.fft.rfft(audio)

frequencies = np.fft.rfftfreq(
    len(audio),
    1 / SAMPLE_RATE
)

magnitudes = np.abs(fft)

print("FFT complete!")

sorted_indices = np.argsort(magnitudes)[::-1]

while True:
    user_input = input(
        "\nHow many frequencies do you want? "
        "(type 'q' to quit): "
    )

    if user_input.lower() == "q":
        break

    try:
        num_frequencies = int(user_input)
    except ValueError:
        print("Please enter a number.")
        continue

    if num_frequencies <= 0:
        print("Enter a number greater than 0. :/")
        continue

    num_frequencies = min(num_frequencies, len(fft))

    filtered_fft = np.zeros_like(fft)

    strongest_indices = sorted_indices[:num_frequencies]

    filtered_fft[strongest_indices] = fft[strongest_indices]

    reconstructed_audio = np.fft.irfft(
        filtered_fft,
        n=len(audio)
    )

    max_amplitude = np.max(np.abs(reconstructed_audio))

    if max_amplitude > 0:
        reconstructed_audio /= max_amplitude

    print(f"Playing using {num_frequencies} frequencies...")

    sd.play(reconstructed_audio, SAMPLE_RATE)

    input("Press ENTER to stop playback...")

    sd.stop()

    print("Playback stopped.")