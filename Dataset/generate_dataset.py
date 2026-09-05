import numpy as np
import pandas as pd
import random
import os
from tqdm import tqdm

# Constants
C = 3e8  # Speed of light
FC = 10e9  # Radar carrier frequency (10 GHz, X-band)
LAMBDA = C / FC
FS = 1000  # Sampling frequency (Hz)
T_SIM = 0.5  # Simulation time (s)
NUM_SAMPLES = int(FS * T_SIM)
NUM_SIMULATIONS = 100000
CHUNK_SIZE = 5000

# Radar elevation angle (rad), e.g., looking slightly upwards
BETA = np.deg2rad(10)

def generate_signal(num_blades, radius, rpm, snr_db):
    t = np.linspace(0, T_SIM, NUM_SAMPLES, endpoint=False)
    omega = 2 * np.pi * rpm / 60.0  # Angular velocity (rad/s)
    
    # Initialize signal
    s = np.zeros(NUM_SAMPLES, dtype=np.complex128)
    
    # Initial phase offset for the rotor
    theta_0 = random.uniform(0, 2*np.pi)
    
    for k in range(num_blades):
        # Phase of the k-th blade
        phi_k = omega * t + theta_0 + k * (2 * np.pi / num_blades)
        
        # Radial velocity component
        v_r = np.cos(phi_k) * np.cos(BETA)
        
        # Sinc term argument: (4*pi/lambda) * (L/2) * cos(phi) * cos(beta)
        # Note: np.sinc in python computes sin(pi*x)/(pi*x), so we divide by pi
        sinc_arg = (4 * np.pi / LAMBDA) * (radius / 2.0) * v_r / np.pi
        
        # Phase term argument: (4*pi/lambda) * (L/2) * cos(phi) * cos(beta)
        phase_arg = (4 * np.pi / LAMBDA) * (radius / 2.0) * v_r
        
        blade_signal = radius * np.sinc(sinc_arg) * np.exp(1j * phase_arg)
        s += blade_signal
        
    # Add AWGN
    # Calculate signal power
    sig_power = np.mean(np.abs(s)**2)
    
    # Calculate noise power based on SNR
    snr_linear = 10 ** (snr_db / 10.0)
    noise_power = sig_power / snr_linear
    
    # Generate complex gaussian noise
    noise = np.sqrt(noise_power / 2) * (np.random.randn(NUM_SAMPLES) + 1j * np.random.randn(NUM_SAMPLES))
    
    s_noisy = s + noise
    return s_noisy

def main():
    output_file = 'helicopter_microdoppler_dataset.csv'
    
    # Generate column names
    cols = ['num_blades', 'radius_m', 'rpm', 'tip_velocity_m_s', 'snr_db']
    for i in range(NUM_SAMPLES):
        cols.append(f'I_{i}')
        cols.append(f'Q_{i}')
        
    # Remove existing file if present
    if os.path.exists(output_file):
        os.remove(output_file)
        
    print(f"Generating {NUM_SIMULATIONS} samples into {output_file}...")
    
    for chunk_idx in tqdm(range(0, NUM_SIMULATIONS, CHUNK_SIZE)):
        data = []
        for _ in range(CHUNK_SIZE):
            # Randomly select a helicopter type
            num_blades = random.choice([2, 3, 4])
            
            if num_blades == 2:
                # E.g., Bell UH-1
                rpm = random.uniform(300, 350)
                radius = random.uniform(6.5, 7.5)
            elif num_blades == 3:
                # E.g., Gazelle
                rpm = random.uniform(360, 400)
                radius = random.uniform(4.5, 5.5)
            else: # 4 blades
                # E.g., AH-64 Apache, UH-60
                rpm = random.uniform(250, 300)
                radius = random.uniform(7.0, 8.5)
                
            tip_velocity = (rpm / 60.0) * 2 * np.pi * radius
            snr_db = random.uniform(5, 25)
            
            s = generate_signal(num_blades, radius, rpm, snr_db)
            
            row = [num_blades, radius, rpm, tip_velocity, snr_db]
            for i in range(NUM_SAMPLES):
                row.append(s[i].real)
                row.append(s[i].imag)
                
            data.append(row)
            
        df = pd.DataFrame(data, columns=cols)
        # Append to CSV
        mode = 'a' if chunk_idx > 0 else 'w'
        header = True if chunk_idx == 0 else False
        df.to_csv(output_file, mode=mode, header=header, index=False)
        
    print("Dataset generation complete.")

if __name__ == "__main__":
    main()
