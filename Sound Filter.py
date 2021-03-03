from numpy import \
    linspace,array, \
    zeros,log,exp,sin,cos,sqrt,pi,e, ones, \
    arange, shape, zeros, real, imag
from matplotlib.pyplot import \
    plot,xlabel,ylabel,legend,show, \
    figure, subplot, title, tight_layout
from scipy.fftpack import fft
import scipy.io.wavfile as wav 
import sounddevice as sd

# time dimension

file_name='Sounds/NorthernCardinal_noise.wav'
Fs, f =wav.read(file_name)
f = f/max(f)

dT =  1/Fs # sec   time between freq samples
nt  = len(f) #   number of samples in record
T =  nt*dT # Time period of record

t=arange(0,T,dT)  #  time array in seconds using arange(start,stop,step)

# frequency dimension

freqf =  1/T # Hz   fundamental frequency (lowest frequency)
nfmax = int(nt/2) # number of frequencies resolved by FFT

freqmax = freqf*nfmax # Max frequency (Nyquist)

freq = arange(0,freqmax,freqf) # frequency array using arange(start,stop,step)


print('Fundamental period and Nyquist Freq',T, freqmax)


# take FFT
F = fft(f)

# get the coeffs
a = 2*real(F[:nfmax])/nt # form the a coefficients
a[0] = a[0]/2

b = -2*imag(F[:nfmax])/nt # form the b coefficients

p = sqrt(a**2 + b**2) # form power spectrum
p_threshold = max(p/2)


fr = ones(nt)*a[0] # fill time series with constant term

# Sum the Fourier Series
Nrecon = nfmax
for n in range(1,Nrecon):
    if p[n] > 0.005:
        fr = fr + a[n]*cos(n/T * 2*pi*t) + b[n]*sin(n/T*2*pi*t)
    
fclean = fr

# write out the clean time series  
wav.write('Clean.wav', Fs, fclean)

# make plots
figure(1)

subplot(2,1,1)
plot(t,f, label='Original')
plot(t,fclean, label='Clean')
title('Signal')
legend()

subplot(2,1,2)

plot(freq, p,'-', label='Power')
legend() 

title('FFT Fourier Coefficients')

tight_layout() 

sd.play(fclean,Fs)