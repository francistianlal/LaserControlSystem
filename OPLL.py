
# -*- coding: utf-8 -*-
"""
Created on Tue 06/15/2023 12:06:02 2023
use for writing the hardware for laser control system
pyro 4 enables you to build applications in which objects can talk to each other over the network, with minimal programming effort

function used:
1. triangular waveform generator
2. triangular waveform function
3. triangular waveform normalizer for noisy signal
4. triangular waveform tranformer for normalize and un-normalize the noisy signal
4. bit controller

@author: Francis Tian
"""

import Pyro4
import time
import pickle
import numpy as np
import pylab as py
import matplotlib.pyplot as plt
from scipy import signal

def generate_triangular_waveform(duration, sampling_freq, frequency, amplitude):
    # Generate time axis
    t = np.linspace(0, duration, int(duration * sampling_freq), endpoint=False)

    # Generate triangular waveform
    triangular_waveform = amplitude * signal.sawtooth(2 * np.pi * frequency * t, width=0.5)

    return t, triangular_waveform

# used to transform the waveform so that the reference and waveform_tbc have the same scale
# the reference_waveform is a normalized MZI beat signal in time domain, obtained by sweeping a perfectly linearly
# FM laser signal across the MZI we used for FM linearity locking, both frequency and amplitude = 1, centered around 0
# Define the triangular wave function
def triangular_wave(t, amplitude, period, center, amplitude_bias):
    return amplitude * (2 / period) * (period / 2 - np.abs((t - center) % period - period / 2)) - amplitude / 2 + amplitude_bias

# use curve fitting principle to guess the paramter for this triangular wave
# then perform the regulation
def regulate_triangular_wave(time, waveform, initial_guess=[1, 1, 1, 1]):
    # Fit the noisy signal with the triangular wave function
    params, params_covariance = curve_fit(triangular_wave, time, waveform, p0=initial_guess)

    # Extract the fitted parameters
    amplitude_fit, period_fit, phase_fit, amplitude_bias_fit = params
    amplitude_fit = abs(amplitude_fit)
    # # Generate the fitted triangular wave signal
    # fitted_signal = triangular_wave(time, amplitude_fit, period_fit, phase_fit, amplitude_bias_fit)
    # plt.plot(time, fitted_signal)

    # signal regulation
    regulated_waveform = (waveform - amplitude_bias_fit) / (amplitude_fit * 0.5)
    regulated_time = time / period_fit
    return regulated_time, regulated_waveform, params

def waveform_transformer(waveform_tbd, time_tbd, reverse = False, para = None):
    # when we are transforming the waveform back to the normal scale for ADC
    # we have to know the para used in normalization and curve fitting
    try:
        amplitude_fit, period_fit, phase_fit, amplitude_bias_fit = para
    except:
        raise KeyError("we have to know the para used in normalization for the program to work")

    if reverse:
        waveform_original = waveform_tbd * (amplitude_fit * 0.5) + amplitude_bias_fit
        time_original = time_tbd * period_fit
        # return the waveform to be used by ADC/DAC
        return waveform_original, time_original
    else:
        regulated_time, regulated_waveform, params = regulate_triangular_wave(time_tbd, waveform_tbd, para)
        # return the normalized waveform
        return regulated_waveform, regulated_time, params

# a PIDController model
class PIDController:
    def __init__(self, kp, ki, kd):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.error_sum = 0
        self.prev_error = 0

    def calculate(self, setpoint, feedback, dt):
        error = setpoint - feedback

        # Proportional term
        p_term = self.kp * error

        # Integral term
        self.error_sum += error * dt
        i_term = self.ki * self.error_sum

        # Derivative term
        d_term = self.kd * (error - self.prev_error) / dt
        self.prev_error = error

        # Total control signal
        control_signal = p_term + i_term + d_term

        return control_signal

def extractSmuggledBits(bDataB, bDataC):
    def sign_extend(value, bits):
        sign_bit = 1 << (bits - 1)
        return (value & (sign_bit - 1)) - (value & sign_bit)

    rawDataB = np.frombuffer(bDataB, dtype='int16')
    rawDataC = np.frombuffer(bDataC, dtype='int16')

    bitS0 = ((rawDataB & (1 << 14)) >> 14).astype('bool')
    bitS1 = ((rawDataB & (1 << 15)) >> 15).astype('bool')
    bitS2 = ((rawDataC & (1 << 14)) >> 14).astype('bool')
    bitS3 = ((rawDataC & (1 << 15)) >> 15).astype('bool')
    bitsS = np.c_[bitS0, bitS1, bitS2, bitS3].T

    dataB = sign_extend(rawDataB & 0x3FFF, 14)
    dataC = sign_extend(rawDataC & 0x3FFF, 14)

    return dataB, dataC, bitsS

Pyro4.config.SERIALIZER = "pickle"
Pyro4.config.PICKLE_PROTOCOL_VERSION = 4

uri = 'PYRO:RedPitaya.Wrapper@rp-f0a610.local:8082'

forceFan = False

print(f'[RP] Connecting to Pyro4 Proxy in \'{uri:s}\'')
oscWrapper = Pyro4.Proxy(uri)
print('[RP] ' + oscWrapper.ping())

# Get calibration scale and offset, and FW Version
currentFWName = oscWrapper.getFWName()
print(f'[RP] RedPitaya running FW version \'{currentFWName:s}\'')

oscWrapper.setChannelsCapture(getA=True, getB=True)

######
# give the reference waveform, this should be imported from a file in the future !!!!!!!!!!!!!
######
reference_waveform = []

# set the ADC/DAC general code
frameTime = 0.1e-3    # Acquisition time in seconds
lineTime = 25e-6  # For camera. Not used
intTime = 21e-6  # For camera. Not used

rpClockT = 8e-9

### Configure Generator, decimation
step0 = 1/4.0
step1 = 1/4.0
amp_max = 1

nBits = 14
maxVal = 2**(nBits-1) - 1
NSamples = int(frameTime / (rpClockT/ step0))
tAxis = np.arange(NSamples) * (rpClockT/ step0)
DAC_scale = maxVal / amp_max
# waveform selection
# sin
f_sin = 1000e3 # Hz
amplitude = 0.1 # V
genWave0 = amplitude * np.sin(2 * np.pi * f_sin * tAxis) * DAC_scale

# triangular
duration = frameTime  # Duration of the waveform in seconds
sampling_freq = step0/rpClockT  # Sampling frequency in Hz
frequency = 10e3 # Frequency of the triangular waveform in Hz
amplitude = 0.1  # Amplitude of the triangular waveform
# Generate the triangular waveform
t, triangular_waveform = generate_triangular_waveform(duration, sampling_freq, frequency, amplitude)
genWave1 = triangular_waveform * DAC_scale

# DC
amplitude = 0.05
genWave2 = amplitude * np.ones(len(tAxis)) * DAC_scale

#plt.figure(1)
#plt.plot(genWave0)
output1 = genWave1
output2 = genWave2

### start DAC waveform generation
oscWrapper.updateGeneratorWaveform(rWaveA = output1, stepA=step0, rWaveB = output2, stepB=step1, syncChannels=True, VERBOSE=False, nbits=14)
oscWrapper.setSignals(fLine=1 / lineTime, laserStatus=False, fanStatus=forceFan, TCamPulse=intTime, TSyncPulse=intTime)

### Configure ADCs
kClockDecimate = 4
oscWrapper.setDecimate(kClockDecimate)
oscWrapper.setNSamples(int(frameTime / 8e-9 / kClockDecimate))

### Start ADCs Acquisition
oscWrapper.startContinuousACQ(startTriggers=False)
oscWrapper.setOutputFrameTrigger(count=None)
print('[RP] Starting to send output frame triggers!!')

### Get Data
bDataB, bDataC, repLen, triggerTS, triggerIndex, wrapped = oscWrapper.getData(timeout=2.0)
dataB, dataC, smuggledBits = extractSmuggledBits(bDataB, bDataC)

ADCBits = 14
ADCRange = 20.0
ADCScale = ADCRange / 2**(ADCBits-1)

input1 = dataB * ADCScale # Control loop
input2 = dataC * ADCScale # Linewidth measurement

#### here set the ADC/DAC to run continuously for 60 second
N_loop = int(60 / frameTime)
loop_1_switch = True
loop_2_switch = False
loop_3_switch = False

for i in range(N_loop):
    print('round ' + str(i))

    ## control loop 1, MZI biasing control
    if loop_1_switch:
        # parameter setting
        # Loop filter (PID controller)
        Kp = 0.1  # Proportional gain
        Ki = 0.01  # Integral gain
        Kd = 0.001  # Derivative gain
        pid = PIDController(kp = Kp, ki = Ki, kd = Kd)
        # Setpoint and initial feedback value
        setpoint = 0.0 # V
        # Simulation time parameters
        dt = frameTime  # Time step
        # Calculate the control signal
        # the feedback signal is the average of min and max of the waveform
        feedback = np.mean([max(input1), min(input1)])
        control_signal = pid.calculate(setpoint, feedback, dt)
        print('here is the first control loop, error signal = ' + str(feedback) + ', control_signal = ' + str(control_signal))
        # apply the control signal to the output
        # DC
        amplitude = amplitude + control_signal
        genWave2 = amplitude * np.ones(len(tAxis)) * DAC_scale

    ## control loop 2, OPLL
    if loop_2_switch:
        # parameter_setting
        f_ref = 1e5
        previous_phase = 0
        last_phase_error = 0

        duration = frameTime  # Duration of the signal in seconds
        sample_rate = 1 / 8e-9 / kClockDecimate  # Number of samples per second
        # gain of PID controller
        Kp = 0.1  # Proportional gain
        Ki = 0.01  # Integral gain
        Kd = 0.001  # Derivative gain

        cutoff_frequency = 100e3  # Cutoff frequency for the low-pass filter in Hz

        K_LVCO = 1e6  # Hz/V
        K_pitaya = 20  # 20mA/V
        K_laser = 200e6  # Hz/mA
        K_VCO = K_LVCO / K_pitaya / K_laser  # VCO gain Hz/V

        amplitude = 0.05  # need to figure out how much voltage is needed

        adc_input = input1
        t = tAxis
        # create the local oscillator signal
        # Generate the LO signal (cosine) with no phase offset
        lo_signal_cos = np.cos(2 * np.pi * f_ref * t)
        # Generate the LO signal (sine) with a 90-degree phase shift
        lo_signal_sin = np.sin(2 * np.pi * f_ref * t)

        # Demodulate the ADC input signal using the LO signals
        I_component = adc_input * lo_signal_cos
        Q_component = adc_input * lo_signal_sin

        # Apply a low-pass filter to the I and Q components
        b, a = signal.butter(4, 2 * cutoff_frequency / sample_rate, 'low')
        I_filtered = signal.lfilter(b, a, I_component)
        Q_filtered = signal.lfilter(b, a, Q_component)

        # Extract the phase using arctan
        phase = -np.arctan(Q_filtered / I_filtered)

        # Apply modulo operation to handle phase wrapping
        # phase alway have to be zero
        phase_wrapped = np.mod(phase + np.pi, 2 * np.pi) - np.pi
        current_phase = phase_wrapped[-1]

        # calculate the phase difference and if no previous phase, skip
        if previous_phase in globals():
            phase_change = current_phase - previous_phase
            # update the phase
            previous_phase = current_phase
        else:
            phase_change = 0
            previous_phase = current_phase

        # Loop filter (PID controller)
        pid = PIDController(kp=Kp, ki=Ki, kd=Kd)
        # Setpoint and initial feedback value
        setpoint = 0.0
        # Simulation time parameters
        dt = frameTime  # Time step
        # Calculate the control signal
        control_signal = pid.calculate(setpoint, phase_change, dt)
        # Apply control signal to triangular wave VCO
        # triangular
        frequency = control_signal * K_VCO  # Frequency of the triangular waveform in Hz
        # Generate the triangular waveform
        t, triangular_waveform = generate_triangular_waveform(duration, sampling_freq, frequency, amplitude)
        genWave1 = triangular_waveform * DAC_scale

    # control loop 3 Active frequency linearization
    if loop_3_switch:
        # calculate the error signal, the waveform here is the beat signal from MZI
        if previous_waveform in globals():
            # normalize the waveform
            current_waveform = waveform_normalizer(input1, reference_waveform)
            waveform_error = current_waveform - previous_waveform
            # update the waveform
            previous_waveform = current_waveform
        else:
            current_waveform = waveform_normalizer(input1, reference_waveform)
            waveform_error = 0
            previous_waveform = current_waveform

        # parameter setting for PID
        # Loop filter (PID controller)
        Kp = 0.1  # Proportional gain
        Ki = 0.01  # Integral gain
        Kd = 0.001  # Derivative gain
        pid = PIDController(kp = Kp, ki = Ki, kd = Kd)
        # Setpoint and initial feedback value
        setpoint = 0.0  # V
        # Simulation time parameters
        dt = frameTime  # Time step
        # Calculate the control signal
        # the feedback signal is the average of min and max of the waveform
        feedback = np.mean([max(input1), min(input1)])
        control_signal = pid.calculate(setpoint, feedback, dt)

    #### ADC/DAC data feeding for next round of control signal generation
    output1 = genWave1
    output2 = genWave2
    ### DAC waveform generation and ADC waveform extraction
    oscWrapper.updateGeneratorWaveform(rWaveA = output1, stepA=step0, rWaveB = output2, stepB=step1, syncChannels=True, VERBOSE=False, nbits=14)
    oscWrapper.setSignals(fLine=1 / lineTime, laserStatus=False, fanStatus=forceFan, TCamPulse=intTime, TSyncPulse=intTime)
    oscWrapper.startContinuousACQ(startTriggers=False)
    bDataB, bDataC, repLen, triggerTS, triggerIndex, wrapped = oscWrapper.getData(timeout=2.0)
    dataB, dataC, smuggledBits = extractSmuggledBits(bDataB, bDataC)
    input1 = dataB * ADCScale # Control loop
    input2 = dataC * ADCScale # Linewidth measurement

### Stop ADCs Acquisition
oscWrapper.stopACQ()
