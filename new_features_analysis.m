close all; clear; clc

load_filename = '/data/alanfr/Desktop/MSc/myAnalysis/All/';


x2_1 = csvread([load_filename '10000L_15.csv']);
x2_2 = csvread([load_filename '10001L_15.csv']);
x2_3 = csvread([load_filename '10010L_15.csv']);
x2_4 = csvread([load_filename '10011L_15.csv']);

x3_1 = csvread([load_filename '10100L_10.csv']);
x3_2 = csvread([load_filename '10101L_10.csv']);
x3_3 = csvread([load_filename '10110L_10.csv']);
x3_4 = csvread([load_filename '10111L_10.csv']);

Fs = 2e6;
%% Feature extraction - compare 4 files

extractor = audioFeatureExtractor("SampleRate",Fs, ...
    "Window",blackmanharris(20000,"periodic"), ...
    "OverlapLength",10000, ...
    "spectralFlatness",true);
setExtractorParams(extractor,"linearSpectrum","SpectrumType","magnitude","WindowNormalization",false);

features_x3_1 = extract(extractor,x3_1);
features_x3_2 = extract(extractor,x3_2);
features_x3_3 = extract(extractor,x3_3);
features_x3_4 = extract(extractor,x3_4);

figure()
h1 = histogram(features_x3_1,50); title('Bebop mode 1'); hold on
h2 = histogram(features_x3_2,50); title('Bebop mode 2')
h3 = histogram(features_x3_3,50); title('Bebop mode 3')
h4 = histogram(features_x3_4,50); title('Bebop mode 4')
legend('Mode 1','Mode 2','Mode 3','Mode 4')

%% Compute overall histogram of a single drone for spectralFlatness

clc;clear;

load_filename = '/data/alanfr/Desktop/MSc/myAnalysis/All/';
Fs = 2e6;

extractor = audioFeatureExtractor("SampleRate",Fs, ...
                                  "Window",hann(20000,"periodic"), ...
                                  "OverlapLength",10000, ...
                                  "spectralFlatness",true);
                             
setExtractorParams(extractor,"linearSpectrum","SpectrumType","magnitude","WindowNormalization",false);


overAllHistogram_Bebop_1 = zeros(1,50);
overAllHistogram_Bebop_2 = zeros(1,50);
overAllHistogram_Bebop_3 = zeros(1,50);
overAllHistogram_Bebop_4 = zeros(1,50);

for i=0:20
    disp(['Opening file -- ',num2str(i)]);
    
    x_Bebop_1 = csvread([load_filename '10000L_',num2str(i),'.csv']);
    features_x_Bebop_1 = extract(extractor,x_Bebop_1);
    h_Bebop_1 = histogram(features_x_Bebop_1,50);
    overAllHistogram_Bebop_1 = overAllHistogram_Bebop_1 + h_Bebop_1.Values;
    
    x_Bebop_2 = csvread([load_filename '10001L_',num2str(i),'.csv']);
    features_x_Bebop_2 = extract(extractor,x_Bebop_2);
    h_Bebop_2 = histogram(features_x_Bebop_2,50);
    overAllHistogram_Bebop_2 = overAllHistogram_Bebop_2 + h_Bebop_2.Values;
    
    x_Bebop_3 = csvread([load_filename '10010L_',num2str(i),'.csv']);
    features_x_Bebop_3 = extract(extractor,x_Bebop_3);
    h_Bebop_3 = histogram(features_x_Bebop_3,50);
    overAllHistogram_Bebop_3 = overAllHistogram_Bebop_3 + h_Bebop_3.Values;
    
    x_Bebop_4 = csvread([load_filename '10011L_',num2str(i),'.csv']);
    features_x_Bebop_4 = extract(extractor,x_Bebop_4);
    h_Bebop_4 = histogram(features_x_Bebop_4,50);
    overAllHistogram_Bebop_4 = overAllHistogram_Bebop_4 + h_Bebop_4.Values;
   
end

a1 = overAllHistogram_Bebop_1(:,1:48);
a2 = overAllHistogram_Bebop_2(:,1:48);
a3 = overAllHistogram_Bebop_3(:,1:48);
a4 = overAllHistogram_Bebop_4(:,1:48);

% plot histograms of Bebop modes over the selected feature
clc;

figure()
subplot(411); bar(a1);
subplot(412); bar(a2); 
subplot(413); bar(a3);
subplot(414); bar(a4);

figure()
bar(a1); hold on
bar(a2);
bar(a3);
bar(a4);
legend('Mode 1','Mode 2','Mode 3','Mode 4')

figure()
subplot(211); bar(a1); hold on; bar(a2);legend('Mode 1','Mode 2')
subplot(212); bar(a3); hold on; bar(a4);legend('Mode 3','Mode 4')

disp(['Bebop mode 1: Mean=',num2str(mean(a1)),' std=',num2str(std(a1))])
disp(['Bebop mode 2: Mean=',num2str(mean(a2)),' std=',num2str(std(a2))])
disp(['Bebop mode 3: Mean=',num2str(mean(a3)),' std=',num2str(std(a3))])
disp(['Bebop mode 4: Mean=',num2str(mean(a4)),' std=',num2str(std(a4))])

mean_a1 = 0;
mean_a2 = 0;
mean_a3 = 0;
mean_a4 = 0;
for i=1:48
    mean_a1 = mean_a1 + a1(i)*i/sum(a1);
    mean_a2 = mean_a2 + a2(i)*i/sum(a2);
    mean_a3 = mean_a3 + a3(i)*i/sum(a3);
    mean_a4 = mean_a4 + a4(i)*i/sum(a4);
end

% plot PDF distributions
figure()
plot(a1); hold on; 
yline(mean(a1),'color','green','lineWidth',2)
yline(mean(a1)+std(a1)/2,'color','black','lineWidth',2)
yline(mean(a1)-std(a1)/2,'color','k','lineWidth',2)

%% Compare all features for one signal

% Output column mapping
% 
%       spectralCentroid: 1
%          spectralCrest: 2
%       spectralDecrease: 3
%        spectralEntropy: 4
%       spectralFlatness: 5
%           spectralFlux: 6
%       spectralKurtosis: 7
%   spectralRolloffPoint: 8
%       spectralSkewness: 9
%          spectralSlope: 10
%         spectralSpread: 11

extractor = audioFeatureExtractor("SampleRate",Fs, ...
    "Window",hamming(20000,"periodic"), ...
    "OverlapLength",10000, ...
    "spectralCentroid",true,"spectralCrest",true, ...
    "spectralDecrease",true,"spectralEntropy",true, ...
    "spectralFlatness",true,"spectralFlux",true, ...
    "spectralKurtosis",true,"spectralRolloffPoint",true, ...
    "spectralSkewness",true,"spectralSlope",true, ...
    "spectralSpread",true);
setExtractorParams(extractor,"linearSpectrum","SpectrumType","magnitude","WindowNormalization",false);
features = extract(extractor,x2_1);

figure()
subplot(431); plot(x2_1);               title('Original signal'); grid on
subplot(432); plot(features(:,1));      title('spectralCentroid'); grid on
subplot(433); plot(features(:,2));      title('spectralCrest'); grid on
subplot(434); plot(features(:,3));      title('spectralDecrease'); grid on
subplot(435); plot(features(:,4));      title('spectralEntropy'); grid on
subplot(436); plot(features(:,5));      title('spectralFlatness'); grid on
subplot(437); plot(features(:,6));      title('spectralFlux'); grid on
subplot(438); plot(features(:,7));      title('spectralKurtosis'); grid on
subplot(439); plot(features(:,8));      title('spectralRolloffPoint'); grid on
subplot(4,3,10); plot(features(:,9));   title('spectralSkewness'); grid on
subplot(4,3,11); plot(features(:,10));  title('spectralSlope'); grid on
subplot(4,3,12); plot(features(:,11));  title('spectralSpread'); grid on

figure()
subplot(431); histogram(x2_1);               title('Original signal'); grid on
subplot(432); histogram(features(:,1));      title('spectralCentroid'); grid on
subplot(433); histogram(features(:,2));      title('spectralCrest'); grid on
subplot(434); histogram(features(:,3));      title('spectralDecrease'); grid on
subplot(435); histogram(features(:,4));      title('spectralEntropy'); grid on
subplot(436); histogram(features(:,5));      title('spectralFlatness'); grid on
subplot(437); histogram(features(:,6));      title('spectralFlux'); grid on
subplot(438); histogram(features(:,7));      title('spectralKurtosis'); grid on
subplot(439); histogram(features(:,8));      title('spectralRolloffPoint'); grid on
subplot(4,3,10); histogram(features(:,9));   title('spectralSkewness'); grid on
subplot(4,3,11); histogram(features(:,10));  title('spectralSlope'); grid on
subplot(4,3,12); histogram(features(:,11));  title('spectralSpread'); grid on

%% compare all features of one drone and all modes

clc;


windowLength = 8e3;
overlap = 4e3;

extractor = audioFeatureExtractor("SampleRate",Fs, ...
    "Window",hamming(windowLength,"periodic"), ...
    "OverlapLength",overlap, ...
    "spectralCentroid",true,"spectralCrest",true, ...
    "spectralDecrease",true,"spectralEntropy",true, ...
    "spectralFlatness",true,"spectralFlux",true, ...
    "spectralKurtosis",true,"spectralRolloffPoint",true, ...
    "spectralSkewness",true,"spectralSlope",true, ...
    "spectralSpread",true);
    setExtractorParams(extractor,"linearSpectrum","SpectrumType","magnitude","WindowNormalization",false);

overAllHistogram_Bebop_mode1 = zeros(11,50);
overAllHistogram_Bebop_mode2 = zeros(11,50);
overAllHistogram_Bebop_mode3 = zeros(11,50);
overAllHistogram_Bebop_mode4 = zeros(11,50);

for i=0:17
    disp(['Opening file -- ',num2str(i)]);
    
    x_Bebop_1 = csvread([load_filename '10000H_',num2str(i),'.csv']);
        features_Bebop_1 = extract(extractor,x_Bebop_1);
        h_Bebop_mode1_feature1  = histogram(features_Bebop_1(:,1),50);  overAllHistogram_Bebop_mode1(1,:)  = overAllHistogram_Bebop_mode1(1,:)  + h_Bebop_mode1_feature1.Values;
        h_Bebop_mode1_feature2  = histogram(features_Bebop_1(:,2),50);  overAllHistogram_Bebop_mode1(2,:)  = overAllHistogram_Bebop_mode1(2,:)  + h_Bebop_mode1_feature2.Values;
        h_Bebop_mode1_feature3  = histogram(features_Bebop_1(:,3),50);  overAllHistogram_Bebop_mode1(3,:)  = overAllHistogram_Bebop_mode1(3,:)  + h_Bebop_mode1_feature3.Values;
        h_Bebop_mode1_feature4  = histogram(features_Bebop_1(:,4),50);  overAllHistogram_Bebop_mode1(4,:)  = overAllHistogram_Bebop_mode1(4,:)  + h_Bebop_mode1_feature4.Values;
        h_Bebop_mode1_feature5  = histogram(features_Bebop_1(:,5),50);  overAllHistogram_Bebop_mode1(5,:)  = overAllHistogram_Bebop_mode1(5,:)  + h_Bebop_mode1_feature5.Values;
        h_Bebop_mode1_feature6  = histogram(features_Bebop_1(:,6),50);  overAllHistogram_Bebop_mode1(6,:)  = overAllHistogram_Bebop_mode1(6,:)  + h_Bebop_mode1_feature6.Values;
        h_Bebop_mode1_feature7  = histogram(features_Bebop_1(:,7),50);  overAllHistogram_Bebop_mode1(7,:)  = overAllHistogram_Bebop_mode1(7,:)  + h_Bebop_mode1_feature7.Values;
        h_Bebop_mode1_feature8  = histogram(features_Bebop_1(:,8),50);  overAllHistogram_Bebop_mode1(8,:)  = overAllHistogram_Bebop_mode1(8,:)  + h_Bebop_mode1_feature8.Values;
        h_Bebop_mode1_feature9  = histogram(features_Bebop_1(:,9),50);  overAllHistogram_Bebop_mode1(9,:)  = overAllHistogram_Bebop_mode1(9,:)  + h_Bebop_mode1_feature9.Values;
        h_Bebop_mode1_feature10 = histogram(features_Bebop_1(:,10),50); overAllHistogram_Bebop_mode1(10,:) = overAllHistogram_Bebop_mode1(10,:) + h_Bebop_mode1_feature10.Values;
        h_Bebop_mode1_feature11 = histogram(features_Bebop_1(:,11),50); overAllHistogram_Bebop_mode1(11,:) = overAllHistogram_Bebop_mode1(11,:) + h_Bebop_mode1_feature11.Values;
    
    x_Bebop_2 = csvread([load_filename '10001H_',num2str(i),'.csv']);
        features_Bebop_2 = extract(extractor,x_Bebop_2);
        h_Bebop_mode2_feature1  = histogram(features_Bebop_2(:,1),50);  overAllHistogram_Bebop_mode2(1,:)  = overAllHistogram_Bebop_mode2(1,:)  + h_Bebop_mode2_feature1.Values;
        h_Bebop_mode2_feature2  = histogram(features_Bebop_2(:,2),50);  overAllHistogram_Bebop_mode2(2,:)  = overAllHistogram_Bebop_mode2(2,:)  + h_Bebop_mode2_feature2.Values;
        h_Bebop_mode2_feature3  = histogram(features_Bebop_2(:,3),50);  overAllHistogram_Bebop_mode2(3,:)  = overAllHistogram_Bebop_mode2(3,:)  + h_Bebop_mode2_feature3.Values;
        h_Bebop_mode2_feature4  = histogram(features_Bebop_2(:,4),50);  overAllHistogram_Bebop_mode2(4,:)  = overAllHistogram_Bebop_mode2(4,:)  + h_Bebop_mode2_feature4.Values;
        h_Bebop_mode2_feature5  = histogram(features_Bebop_2(:,5),50);  overAllHistogram_Bebop_mode2(5,:)  = overAllHistogram_Bebop_mode2(5,:)  + h_Bebop_mode2_feature5.Values;
        h_Bebop_mode2_feature6  = histogram(features_Bebop_2(:,6),50);  overAllHistogram_Bebop_mode2(6,:)  = overAllHistogram_Bebop_mode2(6,:)  + h_Bebop_mode2_feature6.Values;
        h_Bebop_mode2_feature7  = histogram(features_Bebop_2(:,7),50);  overAllHistogram_Bebop_mode2(7,:)  = overAllHistogram_Bebop_mode2(7,:)  + h_Bebop_mode2_feature7.Values;
        h_Bebop_mode2_feature8  = histogram(features_Bebop_2(:,8),50);  overAllHistogram_Bebop_mode2(8,:)  = overAllHistogram_Bebop_mode2(8,:)  + h_Bebop_mode2_feature8.Values;
        h_Bebop_mode2_feature9  = histogram(features_Bebop_2(:,9),50);  overAllHistogram_Bebop_mode2(9,:)  = overAllHistogram_Bebop_mode2(9,:)  + h_Bebop_mode2_feature9.Values;
        h_Bebop_mode2_feature10 = histogram(features_Bebop_2(:,10),50); overAllHistogram_Bebop_mode2(10,:) = overAllHistogram_Bebop_mode2(10,:) + h_Bebop_mode2_feature10.Values;
        h_Bebop_mode2_feature11 = histogram(features_Bebop_2(:,11),50); overAllHistogram_Bebop_mode2(11,:) = overAllHistogram_Bebop_mode2(11,:) + h_Bebop_mode2_feature11.Values;

    x_Bebop_3 = csvread([load_filename '10010H_',num2str(i),'.csv']);
        features_Bebop_3 = extract(extractor,x_Bebop_3);
        h_Bebop_mode3_feature1  = histogram(features_Bebop_3(:,1),50);  overAllHistogram_Bebop_mode3(1,:)  = overAllHistogram_Bebop_mode3(1,:)  + h_Bebop_mode3_feature1.Values;
        h_Bebop_mode3_feature2  = histogram(features_Bebop_3(:,2),50);  overAllHistogram_Bebop_mode3(2,:)  = overAllHistogram_Bebop_mode3(2,:)  + h_Bebop_mode3_feature2.Values;
        h_Bebop_mode3_feature3  = histogram(features_Bebop_3(:,3),50);  overAllHistogram_Bebop_mode3(3,:)  = overAllHistogram_Bebop_mode3(3,:)  + h_Bebop_mode3_feature3.Values;
        h_Bebop_mode3_feature4  = histogram(features_Bebop_3(:,4),50);  overAllHistogram_Bebop_mode3(4,:)  = overAllHistogram_Bebop_mode3(4,:)  + h_Bebop_mode3_feature4.Values;
        h_Bebop_mode3_feature5  = histogram(features_Bebop_3(:,5),50);  overAllHistogram_Bebop_mode3(5,:)  = overAllHistogram_Bebop_mode3(5,:)  + h_Bebop_mode3_feature5.Values;
        h_Bebop_mode3_feature6  = histogram(features_Bebop_3(:,6),50);  overAllHistogram_Bebop_mode3(6,:)  = overAllHistogram_Bebop_mode3(6,:)  + h_Bebop_mode3_feature6.Values;
        h_Bebop_mode3_feature7  = histogram(features_Bebop_3(:,7),50);  overAllHistogram_Bebop_mode3(7,:)  = overAllHistogram_Bebop_mode3(7,:)  + h_Bebop_mode3_feature7.Values;
        h_Bebop_mode3_feature8  = histogram(features_Bebop_3(:,8),50);  overAllHistogram_Bebop_mode3(8,:)  = overAllHistogram_Bebop_mode3(8,:)  + h_Bebop_mode3_feature8.Values;
        h_Bebop_mode3_feature9  = histogram(features_Bebop_3(:,9),50);  overAllHistogram_Bebop_mode3(9,:)  = overAllHistogram_Bebop_mode3(9,:)  + h_Bebop_mode3_feature9.Values;
        h_Bebop_mode3_feature10 = histogram(features_Bebop_3(:,10),50); overAllHistogram_Bebop_mode3(10,:) = overAllHistogram_Bebop_mode3(10,:) + h_Bebop_mode3_feature10.Values;
        h_Bebop_mode3_feature11 = histogram(features_Bebop_3(:,11),50); overAllHistogram_Bebop_mode3(11,:) = overAllHistogram_Bebop_mode3(11,:) + h_Bebop_mode3_feature11.Values;
      
    x_Bebop_4 = csvread([load_filename '10011H_',num2str(i),'.csv']);
        features_Bebop_4 = extract(extractor,x_Bebop_4);
        h_Bebop_mode4_feature1  = histogram(features_Bebop_4(:,1),50);  overAllHistogram_Bebop_mode4(1,:)  = overAllHistogram_Bebop_mode4(1,:)  + h_Bebop_mode4_feature1.Values;
        h_Bebop_mode4_feature2  = histogram(features_Bebop_4(:,2),50);  overAllHistogram_Bebop_mode4(2,:)  = overAllHistogram_Bebop_mode4(2,:)  + h_Bebop_mode4_feature2.Values;
        h_Bebop_mode4_feature3  = histogram(features_Bebop_4(:,3),50);  overAllHistogram_Bebop_mode4(4,:)  = overAllHistogram_Bebop_mode4(3,:)  + h_Bebop_mode4_feature3.Values;
        h_Bebop_mode4_feature4  = histogram(features_Bebop_4(:,4),50);  overAllHistogram_Bebop_mode4(4,:)  = overAllHistogram_Bebop_mode4(4,:)  + h_Bebop_mode4_feature4.Values;
        h_Bebop_mode4_feature5  = histogram(features_Bebop_4(:,5),50);  overAllHistogram_Bebop_mode4(5,:)  = overAllHistogram_Bebop_mode4(5,:)  + h_Bebop_mode4_feature5.Values;
        h_Bebop_mode4_feature6  = histogram(features_Bebop_4(:,6),50);  overAllHistogram_Bebop_mode4(6,:)  = overAllHistogram_Bebop_mode4(6,:)  + h_Bebop_mode4_feature6.Values;
        h_Bebop_mode4_feature7  = histogram(features_Bebop_4(:,7),50);  overAllHistogram_Bebop_mode4(7,:)  = overAllHistogram_Bebop_mode4(7,:)  + h_Bebop_mode4_feature7.Values;
        h_Bebop_mode4_feature8  = histogram(features_Bebop_4(:,8),50);  overAllHistogram_Bebop_mode4(8,:)  = overAllHistogram_Bebop_mode4(8,:)  + h_Bebop_mode4_feature8.Values;
        h_Bebop_mode4_feature9  = histogram(features_Bebop_4(:,9),50);  overAllHistogram_Bebop_mode4(9,:)  = overAllHistogram_Bebop_mode4(9,:)  + h_Bebop_mode4_feature9.Values;
        h_Bebop_mode4_feature10 = histogram(features_Bebop_4(:,10),50); overAllHistogram_Bebop_mode4(10,:) = overAllHistogram_Bebop_mode4(10,:) + h_Bebop_mode4_feature10.Values;
        h_Bebop_mode4_feature11 = histogram(features_Bebop_4(:,11),50); overAllHistogram_Bebop_mode4(11,:) = overAllHistogram_Bebop_mode4(11,:) + h_Bebop_mode4_feature11.Values;
       
end

figure()
subplot(431);    plot(x3_1);  title('Example signal');     grid on
subplot(432);    plot(overAllHistogram_Bebop_mode1(1,:));  hold on; plot(overAllHistogram_Bebop_mode2(1,:));  plot(overAllHistogram_Bebop_mode3(1,:)); plot(overAllHistogram_Bebop_mode4(1,:));      title('spectral Centroid');      grid on; ylim([0 500])
subplot(433);    plot(overAllHistogram_Bebop_mode1(2,:));  hold on; plot(overAllHistogram_Bebop_mode2(2,:));  plot(overAllHistogram_Bebop_mode3(2,:)); plot(overAllHistogram_Bebop_mode4(2,:));      title('spectral Crest');         grid on
subplot(434);    plot(overAllHistogram_Bebop_mode1(3,:));  hold on; plot(overAllHistogram_Bebop_mode2(3,:));  plot(overAllHistogram_Bebop_mode3(3,:)); plot(overAllHistogram_Bebop_mode4(3,:));      title('spectral Decrease');      grid on
subplot(435);    plot(overAllHistogram_Bebop_mode1(4,:));  hold on; plot(overAllHistogram_Bebop_mode2(4,:));  plot(overAllHistogram_Bebop_mode3(4,:)); plot(overAllHistogram_Bebop_mode4(4,:));      title('spectral Entropy');       grid on; ylim([0 500])
subplot(436);    plot(overAllHistogram_Bebop_mode1(5,:));  hold on; plot(overAllHistogram_Bebop_mode2(5,:));  plot(overAllHistogram_Bebop_mode3(5,:)); plot(overAllHistogram_Bebop_mode4(5,:));      title('spectral Flatness');      grid on; ylim([0 500])
subplot(437);    plot(overAllHistogram_Bebop_mode1(6,:));  hold on; plot(overAllHistogram_Bebop_mode2(6,:));  plot(overAllHistogram_Bebop_mode3(6,:)); plot(overAllHistogram_Bebop_mode4(6,:));      title('spectral Flux');          grid on; ylim([0 600])
subplot(438);    plot(overAllHistogram_Bebop_mode1(7,:));  hold on; plot(overAllHistogram_Bebop_mode2(7,:));  plot(overAllHistogram_Bebop_mode3(7,:)); plot(overAllHistogram_Bebop_mode4(7,:));      title('spectral Kurtosis');      grid on; ylim([0 500])
subplot(439);    plot(overAllHistogram_Bebop_mode1(8,:));  hold on; plot(overAllHistogram_Bebop_mode2(8,:));  plot(overAllHistogram_Bebop_mode3(8,:)); plot(overAllHistogram_Bebop_mode4(8,:));      title('spectral RolloffPoint');  grid on; ylim([0 800])
subplot(4,3,10); plot(overAllHistogram_Bebop_mode1(9,:));  hold on; plot(overAllHistogram_Bebop_mode2(9,:));  plot(overAllHistogram_Bebop_mode3(9,:)); plot(overAllHistogram_Bebop_mode4(9,:));      title('spectral Skewness');      grid on; ylim([0 800])
subplot(4,3,11); plot(overAllHistogram_Bebop_mode1(10,:)); hold on; plot(overAllHistogram_Bebop_mode2(10,:)); plot(overAllHistogram_Bebop_mode3(10,:)); plot(overAllHistogram_Bebop_mode4(10,:));    title('spectral Slope');         grid on; ylim([0 800])
subplot(4,3,12); plot(overAllHistogram_Bebop_mode1(11,:)); hold on; plot(overAllHistogram_Bebop_mode2(11,:)); plot(overAllHistogram_Bebop_mode3(11,:)); plot(overAllHistogram_Bebop_mode4(11,:));    title('spectral Spread');        grid on; ylim([0 1000]); legend('Mode 1', 'Mode 2', 'Mode 3', 'Mode 4')

%% Compare effect of window length and window type

signal = x2_1;
fs = 2e6;
numCoeff = 4;

subplot(3,3,1); 
windowLength = 15e3;    
overlapLength = 5e3;
win = hamming(windowLength,"periodic");
S = stft(signal,"Window",win,"OverlapLength",overlapLength,'centered',true);
coeffs = mfcc(S,fs,"NumCoeffs",numCoeff);
plot(coeffs); title(['Hamming, length=',num2str(windowLength),", overlap=",num2str(overlapLength)])

subplot(3,3,2); 
windowLength = 8e3;    overlapLength = 4e3;
win = hamming(windowLength,"periodic");
S = stft(signal,"Window",win,"OverlapLength",overlapLength);
coeffs = mfcc(S,fs,"NumCoeffs",numCoeff);
plot(coeffs); title(['Hamming, length=',num2str(windowLength),", overlap=",num2str(overlapLength)])

subplot(3,3,3); 
windowLength = 4e3;    overlapLength = 2e3;
win = hamming(windowLength,"periodic");
S = stft(signal,"Window",win,"OverlapLength",overlapLength,"Centered",true);
coeffs = gtcc(S,fs,"NumCoeffs",numCoeff);
plot(coeffs); title(['Hamming, length=',num2str(windowLength),", overlap=",num2str(overlapLength)])

subplot(3,3,4); 
windowLength = 20e3;    overlapLength = 10e3;
win = hann(windowLength,"periodic");
S = stft(signal,"Window",win,"OverlapLength",overlapLength,"Centered",true);
coeffs = gtcc(S,fs,"NumCoeffs",numCoeff);
plot(coeffs); title(['Hann, length=',num2str(windowLength),", overlap=",num2str(overlapLength)])

subplot(3,3,5); 
windowLength = 8e3;    overlapLength = 4e3;
win = hann(windowLength,"periodic");
S = stft(signal,"Window",win,"OverlapLength",overlapLength,"Centered",true);
coeffs = gtcc(S,fs,"NumCoeffs",numCoeff);
plot(coeffs); title(['Hann, length=',num2str(windowLength),", overlap=",num2str(overlapLength)])

subplot(3,3,6); 
windowLength = 4e3;    overlapLength = 2e3;
win = hann(windowLength,"periodic");
S = stft(signal,"Window",win,"OverlapLength",overlapLength,"Centered",true);
coeffs = gtcc(S,fs,"NumCoeffs",numCoeff);
plot(coeffs); title(['Hann, length=',num2str(windowLength),", overlap=",num2str(overlapLength)])

subplot(3,3,7); 
windowLength = 20e3;    overlapLength = 10e3;
win = blackmanharris(windowLength,"periodic");
S = stft(signal,"Window",win,"OverlapLength",overlapLength,"Centered",true);
coeffs = gtcc(S,fs,"NumCoeffs",numCoeff);
plot(coeffs); title(['BlackmanHaris, length=',num2str(windowLength),", overlap=",num2str(overlapLength)])

subplot(3,3,8); 
windowLength = 8e3;    overlapLength = 4e3;
win = blackmanharris(windowLength,"periodic");
S = stft(signal,"Window",win,"OverlapLength",overlapLength,"Centered",true);
coeffs = gtcc(S,fs,"NumCoeffs",numCoeff);
plot(coeffs); title(['BlackmanHaris, length=',num2str(windowLength),", overlap=",num2str(overlapLength)])

subplot(3,3,9); 
windowLength = 4e3;    overlapLength = 2e3;
win = blackmanharris(windowLength,"periodic");
S = stft(signal,"Window",win,"OverlapLength",overlapLength,"Centered",true);
coeffs = gtcc(S,fs,"NumCoeffs",numCoeff);
plot(coeffs); title(['BlackmanHaris, length=',num2str(windowLength),", overlap=",num2str(overlapLength)])

%%
numCoeff = 4;
i = 1;
for overlapLength=0:500:4e3
    windowLength = 8e3;    
    win = hamming(windowLength,"periodic");
    S = stft(signal,"Window",win,"OverlapLength",overlapLength);
    coeffs = gtcc(S,fs,"NumCoeffs",numCoeff);
    subplot(3,3,i)
    plot(coeffs); 
    grid on
    title(["overlap=",num2str(overlapLength)])
    i = i+1;
end