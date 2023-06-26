
# -*- coding: utf-8 -*-
"""
Created on Tue 06/15/2023 12:06:02 2023
use for writing the hardware for laser control system
pyro 4 enables you to build applications in which objects can talk to each other over the network, with minimal programming effort

function used:
1. triangular waveform generator
2. triangular waveform function
3. triangular waveform normalizer for noisy signal
4. triangular waveform transformer for normalize and un-normalize the noisy signal
5. frequency shifter for triangular wave
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

# waveform error calculator, both waveform must be regulated
def error_calculator(current_waveform, regulated_time, reference_waveform):
    # Extend and mirror the ramp signal to match the length and shape of the triangular wave signal
    num_points = len(reference_waveform) * 2
    t_reference = np.linspace(0, 1, num_points + 1)[:-1]
    extended_ramp_signal = np.concatenate((reference_waveform, np.flip(reference_waveform)))

    # Repeat and trim the extended ramp signal to match the length of the triangular wave signal
    repitition_number = max(regulated_time) + 1
    extended_ramp_signal_matched = np.tile(extended_ramp_signal, int(repitition_number))

    # find the first minima of the triangular wave signal in the time domain
    min_index = None
    for i in range(1, len(current_waveform) - 1):
        if current_waveform[i] < current_waveform[i - 1] and current_waveform[i] < current_waveform[i + 1]:
            min_index = i
            min_time = regulated_time[i]
            break

    start_point = 0
    num_steps = np.mean(np.diff(t_reference))
    num_samples = len(extended_ramp_signal_matched)
    end = start_point + (num_steps * num_samples)
    extended_reference_time = np.linspace(start_point, end, num_samples)

    shift_time = 1 - min_time
    extended_reference_time = extended_reference_time - shift_time

    # now we calculate the error by calculating the difference between reference and current waveform
    waveform_error_storage = []
    for index, value in enumerate(current_waveform):
        time = regulated_time[index]
        # find the corresponding value in the extended ramp
        i_t_ref = np.abs(extended_reference_time - time).argmin()
        # calculate the error
        waveform_error_unit = value - extended_ramp_signal_matched[i_t_ref]
        waveform_error_storage.append(waveform_error_unit)
    # waveform error is an array with the same length as the current waveform, which will be fed to the output DAC
    return waveform_error_storage

# waveform frequency shifter
def change_frequency(waveform, original_frequency, target_frequency):
    original_period = 1 / original_frequency
    target_period = 1 / target_frequency

    original_samples = len(waveform)
    target_samples = original_samples

    original_time = np.arange(original_samples) * original_period
    target_time = np.linspace(0, target_period * target_samples, target_samples, endpoint=False)

    resampled_waveform = np.interp(target_time, original_time, waveform)

    return resampled_waveform, target_frequency

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
# give the reference waveform, should be rising ramp with a length of 0.5 sec
# , this should be imported from a file in the future !!!!!!!!!!!!!
######
reference_waveform = []

# set the ADC/DAC general code
frameTime = 0.1e-3    # Acquisition time in seconds
lineTime = 25e-6  # For camera. Not used
intTime = 21e-6  # For camera. Not used
kClockDecimate = 2
rpClockT = 8e-9

### Configure Generator, decimation
step0 = 1/kClockDecimate
step1 = 1/kClockDecimate
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
amplitude = 0.001  # Amplitude of the triangular waveform
# Generate the triangular waveform
t, triangular_waveform = generate_triangular_waveform(duration, sampling_freq, frequency, amplitude)
genWave1 = triangular_waveform * DAC_scale

# DC
DC_amplitude = 0.037
genWave2 = DC_amplitude * np.ones(len(tAxis)) * DAC_scale

#plt.figure(1)
#plt.plot(genWave0)
output1 = genWave1
output2 = genWave2

### start DAC waveform generation
oscWrapper.updateGeneratorWaveform(rWaveA = output1, stepA=step0, rWaveB = output2, stepB=step1, syncChannels=True, VERBOSE=False, nbits=14)
oscWrapper.setSignals(fLine=1 / lineTime, laserStatus=False, fanStatus=forceFan, TCamPulse=intTime, TSyncPulse=intTime)

### Configure ADCs
oscWrapper.setDecimate(kClockDecimate)
oscWrapper.setNSamples(int(frameTime / (rpClockT * kClockDecimate)))

### Start ADCs Acquisition
oscWrapper.startContinuousACQ(startTriggers=False)
oscWrapper.setOutputFrameTrigger(count=None)
print('[RP] Starting to send output frame triggers!!')

### Get Data
bDataB, bDataC, repLen, triggerTS, triggerIndex, wrapped = oscWrapper.getData(timeout=2)
dataB, dataC, smuggledBits = extractSmuggledBits(bDataB, bDataC)

ADCBits = 14
ADCRange = 20.0
ADCScale = ADCRange / 2**(ADCBits-1)

input1 = dataB * ADCScale # Control loop
input2 = dataC * ADCScale # Linewidth measurement

plt.plot(tAxis, input1)
plt.plot(t, triangular_waveform*1e3)
#### here set the ADC/DAC to run continuously for 60 second
N_loop = int(30 / frameTime)
loop_1_switch = True
loop_2_switch = False
loop_3_switch = False

for i in range(N_loop):
    print('round ' + str(i))

    ## control loop 1, MZI biasing control
    if loop_1_switch:
        # parameter setting
        # Loop filter (PID controller)
        Kp = 1e-11  # Proportional gain
        Ki = 0  # Integral gain
        Kd = 1e-7  # Derivative gain
        pid = PIDController(kp = Kp, ki = Ki, kd = Kd)
        # Setpoint and initial feedback value
        setpoint = 0 # V
        # Simulation time parameters
        dt = frameTime  # Time step
        # Calculate the control signal
        # the feedback signal is the average of min and max of the waveform
        feedback = np.mean([max(input1), min(input1)])
#        if 1:
#            plt.plot(tAxis, input1)
#            regulated_time, regulated_waveform, params = regulate_triangular_wave(tAxis, input1,
 #                                                                                 initial_guess=[1, 1, 1, 1])

#            break

        print('max = ' + str(max(input1)) + ', min = ' + str(min(input1)))
        control_signal = pid.calculate(setpoint, feedback, dt)
        # apply the control signal to the output
        # DC
        DC_amplitude = DC_amplitude - control_signal
        print('here is the first control loop, error signal = ' + str(feedback) + ', DC_amplitude = ' + str(DC_amplitude))


        genWave2 = DC_amplitude * np.ones(len(tAxis)) * DAC_scale
        if DC_amplitude < 0.04 or DC_amplitude > 0.06:
            break

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
        # if control loop 3 is on, update the frequency only, otherwise update the waveform
        if not loop_3_switch:
            # Generate the triangular waveform
            t, triangular_waveform = generate_triangular_waveform(duration, sampling_freq, frequency, amplitude)
            genWave1 = triangular_waveform * DAC_scale

    # control loop 3 Active frequency linearization
    if loop_3_switch:
        # calculate the error signal, the waveform here is the beat signal from MZI
        time_tbd = tAxis
        waveform_tbd = input1
        reverse = False
        ### this can be commented after parameter is extracted
        N_sample_visual = 1000
        plt.plot(time_tbd[1:N_sample_visual], waveform_tbd[1:N_sample_visual])
        ###
        para = [] # amplitude, period, phase, amplitude_bias
        # regulate the waveform from ADC
        current_waveform, regulated_time, para = waveform_transformer(waveform_tbd, time_tbd, reverse = False, para = para)

        if previous_waveform in globals():
            # calculate the waveform error for updating the output AC
            waveform_error = error_calculator(current_waveform, regulated_time, reference_waveform)
            # the waveform error will be reverse back to original
            waveform_error_original, time_original = waveform_transformer(waveform_error, regulated_time, reverse=True, para=para)
            # update the waveform
            previous_waveform = current_waveform
        else:
            waveform_error_original = 0
            previous_waveform = current_waveform

        # parameter setting for PID
        # Loop filter (PID controller)
        Kp = 0.1  # Proportional gain
        Ki = 0.01  # Integral gain
        Kd = 0.001  # Derivative gain
        pid = PIDController(kp = Kp, ki = Ki, kd = Kd)
        # Setpoint and initial feedback value
        setpoints = np.zeros(int(len(waveform_error_original)))  # V

        # Simulation time parameters
        dt = frameTime  # Time step
        # Calculate the control signal
        # the feedback signal is a waveform with the same length of input
        feedbacks = waveform_error_original
        control_signal = pid.calculate(setpoints, feedbacks, dt)

        # generates the output signal
        if previous_genWave1 in globals():
            # updating the output waveform
            current_genWave1 = previous_genWave1 - control_signal
            # this is to implement the OPLL, change the frequency of the output sigal together with the waveform
            frequency_shifted_waveform = change_frequency(current_genWave1, previous_frequency, frequency)
            current_genWave1 = frequency_shifted_waveform
            # store the changes
            previous_genWave1 = current_genWave1
            previous_frequency = frequency

        else:
            # Generate the triangular waveform
            t, triangular_waveform = generate_triangular_waveform(duration, sampling_freq, frequency, amplitude)
            current_genWave1 = triangular_waveform
            # store the changes
            previous_genWave1 = current_genWave1
            previous_frequency = frequency

        genWave1 = current_genWave1 * DAC_scale

    #### ADC/DAC data feeding for next round of control signal generation
    output1 = genWave1
    output2 = genWave2
    ### DAC waveform generation and ADC waveform extraction
    oscWrapper.updateGeneratorWaveform(rWaveA = output1, stepA=step0, rWaveB = output2, stepB=step1, syncChannels=True, VERBOSE=False, nbits=14)
    oscWrapper.setSignals(fLine=1 / lineTime, laserStatus=False, fanStatus=forceFan, TCamPulse=intTime, TSyncPulse=intTime)

    bDataB, bDataC, repLen, triggerTS, triggerIndex, wrapped = oscWrapper.getData(timeout=2.0)
    dataB, dataC, smuggledBits = extractSmuggledBits(bDataB, bDataC)
    input1 = dataB * ADCScale # Control loop
    input2 = dataC * ADCScale # Linewidth measurement

### Stop ADCs Acquisition
oscWrapper.stopACQ()