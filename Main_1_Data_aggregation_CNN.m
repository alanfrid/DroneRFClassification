close all; clear;clc
load_filename = '/data/alanfr/Desktop/MSc/myAnalysis/All/';  % Path of raw RF data
save_filename = '/data/alanfr/Desktop/MSc/myAnalysis/results/CNN - 29.4.2022/';   % Path of aggregated data

%% Parameters
BUI{1,1} = {'00000'};                         % BUI of the background     RF activities
BUI{1,2} = {'10000','10001','10010','10011'}; % BUI of the Bebop    drone RF activities
BUI{1,3} = {'10100','10101','10110','10111'}; % BUI of the AR       drone RF activities
BUI{1,4} = {'11000'};                         % BUI of the Phantom  drone RF activities

fs = 2e6;                                     % Samples per 1 Second = Sampling Frequency
windowLength = 8e3;                           % Spectral window length
overlapLength = 4e3;                          % Spectral window overlap
numCoeff = 4;                                 % Number of MFCC/GTCC coefficients
FFTLength = 8e3;                              % STFT window length in time domain
win = hamming(windowLength,"periodic");       % Spectral window type

%% Explor STFT parameters
clc;
S = stft(x,fs,...
        "Window",win,...
        "OverlapLength",overlapLength,...
        "FFTLength",FFTLength);
mfcc(S,fs,...
        "Window",win,...
        "OverlapLength",overlapLength,...
        "FFTLength",FFTLength);
    
% Gives 2499 results, of length 8000 frequency samples    
% Results number = floor[ (length(x) - overlapLength)/(windowLength - overlapLength) ]
plot(abs(S))


%% Main

cnt = 1;
data_fin = [];
for opt = 1:length(BUI)
    for b = 1:length(BUI{1,opt})
        disp('starting BUI: ' + string(BUI{1,opt}{b}))
        if(strcmp(BUI{1,opt}{b},'00000'))
            N = 40; 
        elseif(strcmp(BUI{1,opt}{b},'10111'))
            N = 17;
        else
            N = 20; 
        end
        
        for n = 0:N
            % 1. Data import
            disp('Loading files...')
            x = csvread([load_filename BUI{1,opt}{b} 'L_' num2str(n) '.csv']);
            S = stft(x,"Window",win,"OverlapLength",overlapLength); % default FFTlength 128
            coeffs = gtcc(S,fs,"NumCoeffs",numCoeff);
            
            % 2. Data aggregation:
            data = cat(1, cat(1, cat(1, coeffs(:,1),...
                                        coeffs(:,2)),...
                                        coeffs(:,3)),...
                                        coeffs(:,4));
            
            target = [];
            switch(BUI{1,opt}{b})
                case '00000' % Noise
                    target = [1;1;1];
					
                case '10000' % Bebop 1
                    target = [2;2;2];
					
                case '10001' % Bebop 2
					target = [2;2;3];
					
                case '10010' % Bebop 3
					target = [2;2;4];
					
                case '10011' % Bebop 4
					target = [2;2;5];
					
				case '10100' % AR 1
					target = [2;3;6];
					
				case '10101' % AR 2
					target = [2;3;7];
					
				case '10110' % AR 3
					target = [2;3;8];
					
				case '10111' % AR 4
					target = [2;3;9];
					
				case '11000' % Phantom
					target = [2;4;10];
            end
            
            data = cat(1, data, target);
            data_fin = cat(2, data_fin, data);
            
            cnt = cnt + 1;
            disp(string(100*n/N) + '%')
            
        end
    end
end
%%
disp('Saving to data file')
csvwrite([save_filename 'data_CNN_fixed.csv'],data_fin);
disp('Done.')
