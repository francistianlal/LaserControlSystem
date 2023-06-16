
# pyro 4 enables you to build applications in which objects can talk to each other over the network, with minimal programming effort
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

# waveform selection
# sin
f_sin = 1000e3
amplitude = 0.1
genWave0 = maxVal * np.sin(2 * np.pi * f_sin * tAxis) * (amplitude / amp_max)

# DC
amplitude = 0
genWave0 = maxVal * np.ones(len(tAxis)) * (amplitude / amp_max)

# triangular
duration = frameTime  # Duration of the waveform in seconds
sampling_freq = step0/rpClockT  # Sampling frequency in Hz
frequency = 10e3 # Frequency of the triangular waveform in Hz
amplitude = 0.1  # Amplitude of the triangular waveform
# Generate the triangular waveform
t, triangular_waveform = generate_triangular_waveform(duration, sampling_freq, frequency, amplitude)
genWave0 = maxVal * triangular_waveform

#plt.figure(1)
#plt.plot(genWave0)

oscWrapper.updateGeneratorWaveform(rWaveA=genWave0, stepA=step0, rWaveB=genWave0, stepB=step1, syncChannels=True, VERBOSE=False, nbits=14)
oscWrapper.setSignals(fLine=1 / lineTime, laserStatus=False, fanStatus=forceFan, TCamPulse=intTime, TSyncPulse=intTime)

### Configure ADCs
kClockDecimate = 8
oscWrapper.setDecimate(kClockDecimate)
oscWrapper.setNSamples(int(frameTime / 8e-9 / kClockDecimate))

### Configure PID Settings
# Channel 11
oscWrapper.pidSetSetpoint(rawValue=0, ch='11')
oscWrapper.pidSetParams(rawP=0, rawI=0, rawD=0, ch='11')

# Channel 12
oscWrapper.pidSetSetpoint(rawValue=0, ch='12')
oscWrapper.pidSetParams(rawP=0, rawI=0, rawD=0, ch='12')

# Channel 21
oscWrapper.pidSetSetpoint(rawValue=0, ch='21')
oscWrapper.pidSetParams(rawP=0, rawI=0, rawD=0, ch='21')

# Channel 22
oscWrapper.pidSetSetpoint(rawValue=0, ch='22')
oscWrapper.pidSetParams(rawP=0, rawI=0, rawD=0, ch='22')

# And reset all integrators
oscWrapper.pidResetIntegrator(int11=True,int12=True,int21=True,int22=True)

### Start ADCs Acquisition
oscWrapper.startContinuousACQ(startTriggers=False)
oscWrapper.setOutputFrameTrigger(count=None)
print('[RP] Starting to send output frame triggers!!')

### Get Data
bDataB, bDataC, repLen, triggerTS, triggerIndex, wrapped = oscWrapper.getData(timeout=2.0)
dataB, dataC, smuggledBits = extractSmuggledBits(bDataB, bDataC)

ADCBits  = 14
ADCRange = 20.0

ADCScale = ADCRange / 2**(ADCBits-1)

### Stop ADCs Acquisition
oscWrapper.stopACQ()
oscWrapper.stopACQ()