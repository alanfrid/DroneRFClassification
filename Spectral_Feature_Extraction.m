 %% 1.a) Import data
close all; clear; clc

load_filename = 'E:\Msc\Research\My Analysis\Databases\RF\All\';

% Background noise
% x1 = csvread([load_filename '00000L_0.csv']);
% y1 = csvread([load_filename '00000H_0.csv']);
% 
% % Bebop
x2_1 = csvread([load_filename '10000L_0.csv']);
% y2_1 = csvread([load_filename '10000H_0.csv']);
% x2_2 = csvread([load_filename '10001L_0.csv']);
% y2_2 = csvread([load_filename '10001H_0.csv']);
% x2_3 = csvread([load_filename '10010L_0.csv']);
% y2_3 = csvread([load_filename '10010H_0.csv']);
% x2_4 = csvread([load_filename '10011L_0.csv']);
% y2_4 = csvread([load_filename '10011H_0.csv']);
% 
% % AR
x3_1 = csvread([load_filename '10100L_0.csv']);
% y3_1 = csvread([load_filename '10100H_0.csv']);
% x3_2 = csvread([load_filename '10101L_0.csv']);
% y3_2 = csvread([load_filename '10101H_0.csv']);
% x3_3 = csvread([load_filename '10110L_0.csv']);
% y3_3 = csvread([load_filename '10110H_0.csv']);
% x3_4 = csvread([load_filename '10111L_0.csv']);
% y3_4 = csvread([load_filename '10111H_0.csv']);
% 
% %Phantom
x4 = csvread([load_filename '11000L_0.csv']);
% y4 = csvread([load_filename '11000H_0.csv']);

L = length(x2_1);       % Total number samples in each file
T = 5;                  % Total duration of each file [Seconds]
Fs = L/T;               % Samples per 1 Second = Sampling Frequency
t = linspace(1,T,L);    % Time domain span vector

%% CWT zoom-in to preamble

x2_1_short = x2_1(3846480:3850220);
x3_1_short = x3_1(5127246:5129670);

preamble = x2_1_short(1:700);
final = x2_1_short(3500:end);

%%

% Create and set up an audioFeatureExtractor object
extractor = audioFeatureExtractor("SampleRate",Fs, ...
    "Window",blackmanharris(4096,"periodic"), ...
    "OverlapLength",round(4096*0.5), ...
    "SpectralDescriptorInput","melSpectrum", ...
    "linearSpectrum",true,"melSpectrum",true, ...
    "barkSpectrum",true,"erbSpectrum",true, ...
    "mfcc",true,"mfccDelta",true, ...
    "gtcc",true,"gtccDelta",true, ...
    "spectralCentroid",true,"spectralCrest",true, ...
    "spectralDecrease",true,"spectralEntropy",true, ...
    "spectralFlatness",true,"spectralFlux",true, ...
    "spectralKurtosis",true,"spectralRolloffPoint",true, ...
    "spectralSkewness",true,"spectralSlope",true, ...
    "spectralSpread",true,"harmonicRatio",true);

% Extract features from audio data
features = extract(extractor,x3_1);

%% Feature vector desctiption:

%    Output column mapping
% 
%             linearSpectrum: 1:2049
%                melSpectrum: 2050:2081
%               barkSpectrum: 2082:2113
%                erbSpectrum: 2114:2191
%                       mfcc: 2192:2204
%                  mfccDelta: 2205:2217
%                       gtcc: 2218:2230
%                  gtccDelta: 2231:2243
%           spectralCentroid: 2244
%              spectralCrest: 2245
%           spectralDecrease: 2246
%            spectralEntropy: 2247
%           spectralFlatness: 2248
%               spectralFlux: 2249
%           spectralKurtosis: 2250
%       spectralRolloffPoint: 2251
%           spectralSkewness: 2252
%              spectralSlope: 2253
%             spectralSpread: 2254
%              harmonicRatio: 2255

%%

figure(1)
subplot(511); plot(x2_1); title('Original signal - Bebop - mode 1 - Low band')
subplot(512); plot(features(:,1:2049)); title('linearSpectrum')
subplot(513); plot(features(:,2050:2081)); title('melSpectrum')
subplot(514); plot(features(:,2082:2113)); title('barkSpectrum')
subplot(515); plot(features(:,2114:2191)); title('erbSpectrum')

figure(2)
subplot(311); plot(x2_1); title('Original signal - Bebop - mode 1 - Low band')
subplot(312); plot(features(:,2192:2204)); title('MFCC')
subplot(313); plot(features(:,2218:2230)); title('GTCC')

figure(3)
subplot(411); plot(x2_1); title('Original signal - Bebop - mode 1 - Low band')
subplot(412); plot(features(:,2244)); title('spectralCentroid')
subplot(413); plot(features(:,2245)); title('spectralCrest')
subplot(414); plot(features(:,2246)); title('spectralDecrease')

figure(4)
subplot(411); plot(x2_1); title('Original signal - Bebop - mode 1 - Low band')
subplot(412); plot(features(:,2247)); title('spectralEntropy')
subplot(413); plot(features(:,2248)); title('spectralFlatness')
subplot(414); plot(features(:,2249)); title('spectralFlux')

figure(5)
subplot(411); plot(x2_1); title('Original signal - Bebop - mode 1 - Low band')
subplot(412); plot(features(:,2250)); title('spectralKurtosis')
subplot(413); plot(features(:,2251)); title('spectralRolloffPoint')
subplot(414); plot(features(:,2252)); title('spectralSkewness')

figure(6)
subplot(411); plot(x2_1); title('Original signal - Bebop - mode 1 - Low band')
subplot(412); plot(features(:,2253)); title('spectralSlope')
subplot(413); plot(features(:,2254)); title('spectralSpread')
subplot(414); plot(features(:,2255)); title('harmonicRatio')