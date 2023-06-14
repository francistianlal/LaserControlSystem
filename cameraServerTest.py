# pyro 4 enables you to build applications in which objects can talk to each other over the network, with minimal programming effort
import Pyro4
import time
import pickle
import numpy as np
import pylab as py

import matplotlib
matplotlib.use('Qt5Agg')


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

uri = 'PYRO:RedPitaya.Wrapper@169.254.239.95:8082'

forceFan = False

print(f'[RP] Connecting to Pyro4 Proxy in \'{uri:s}\'')
oscWrapper = Pyro4.Proxy(uri)
print('[RP] ' + oscWrapper.ping())

# Get calibration scale and offset, and FW Version
currentFWName = oscWrapper.getFWName()
print(f'[RP] RedPitaya running FW version \'{currentFWName:s}\'')

oscWrapper.setChannelsCapture(getA=True, getB=True)

frameTime = 0.1    # Acquisition time in seconds
lineTime = 25e-6  # For camera. Not used
intTime = 21e-6  # For camera. Not used

### Configure Generator, decimation
step0 = 1/8.0
step1 = 1/8.0

genWave0 = np.arange(16384, dtype='int16')
#genWave1 = -1*np.arange(16384, dtype='int16')

oscWrapper.updateGeneratorWaveform(rWaveA=genWave0, stepA=step0, rWaveB=-1*genWave0, stepB=step1, syncChannels=True, VERBOSE=False, nbits=16)
oscWrapper.setSignals(fLine=1 / lineTime, laserStatus=False, fanStatus=forceFan, TCamPulse=intTime, TSyncPulse=intTime)

### Configure ADCs
kClockDecimate = 32
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

### Stop ADCs Acquisition
oscWrapper.stopACQ()
