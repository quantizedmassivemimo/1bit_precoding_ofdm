function precoder_sim_ofdm(varargin)
% =========================================================================
% Simulator for "Nonlinear Precoding for Phase-Quantized Constant-Envelope
% Massive MU-MIMO-OFDM"
% -------------------------------------------------------------------------
% Revision history:
%   - may-02-2018  v0.1   sj: simplified/commented code for GitHub
% -------------------------------------------------------------------------
% (c) 2018 Sven Jacobsson and Christoph Studer 
% e-mail: sven.jacobsson@ericsson.com and studer@cornell.edu 
% -------------------------------------------------------------------------
% If you this simulator or parts of it, then you must cite our paper:
%   -- S. Jacobsson, O. Castaneda, C. Jeon, G. Durisi, and C. Studer, 
%   "Nonlinear precoding for phase-quantized constant-envelope massive 
%   MU-MIMO-OFDM," in IEEE Int. Conf. Telecommunications (ICT), Saint-Malo, 
%   France, Jun. 2018, to appear.
%=========================================================================

    % set up default/custom parameters
    if isempty(varargin)
        
        disp('using default simulation settings and parameters...');

        % set default simulation parameters
        par.runId = 1; % simulation ID (used to reproduce results)
        par.plot = true; % plot results (true,false)
        par.save = false; % save results (true,false)
        par.P = 2; % number of phase bits: 1, 2, 3, Inf 
        par.U = 16; % number of UEs
        par.B = 128; % number of BS antenns
        par.S = 600; % number of occupied subcarriers: 72, 144, 300, 600, 1200
        par.T = 10; % number of channel taps
        par.sampling_rate_multiplier = 1; % sampling rate multiplier
        par.trials = 10; % number of Monte-Carlo trials (transmissions)
        par.SNRdB_list = -5:2.5:15; % list of SNR [dB] values to be simulated
        par.mod = '16QAM'; % modulation type: 'BPSK', 'QPSK', '8PSK', '16QAM','64QAM'
        par.precoder = {'ZF', 'SQUID', 'ZF_inf'}; % select precoding scheme(s) to be evaluated

    else

        % use custom simulation parameters
        disp('use custom simulation settings and parameters...')
        par = varargin{1}; % load custom simulation parameters

    end

    % -- initialization
    
    % set unique filename
    par.simName = ['BER_',num2str(par.U),'x',num2str(par.B),'_',num2str(par.mod),...
        '_',num2str(par.runId),'_',datestr(clock,0)];
    
    % use runId random seed (enables reproducibility)
    rng(par.runId);

    % set up Gray-mapped constellation alphabets
    switch (par.mod)
        case 'BPSK'
            par.symbols = [ -1 1 ];
        case 'QPSK'
            par.symbols = [ -1-1i,-1+1i,+1-1i,+1+1i ];
        case '8PSK'
            par.symbols = [...
                exp(1i*2*pi/8*0), exp(1i*2*pi/8*1), ...
                exp(1i*2*pi/8*7), exp(1i*2*pi/8*6), ...
                exp(1i*2*pi/8*3), exp(1i*2*pi/8*2), ...
                exp(1i*2*pi/8*4), exp(1i*2*pi/8*5)];
        case '16QAM'
            par.symbols = [...
                -3-3i,-3-1i,-3+3i,-3+1i, ...
                -1-3i,-1-1i,-1+3i,-1+1i, ...
                +3-3i,+3-1i,+3+3i,+3+1i, ...
                +1-3i,+1-1i,+1+3i,+1+1i ];
        case '64QAM'
            par.symbols = [...
                -7-7i,-7-5i,-7-1i,-7-3i,-7+7i,-7+5i,-7+1i,-7+3i, ...
                -5-7i,-5-5i,-5-1i,-5-3i,-5+7i,-5+5i,-5+1i,-5+3i, ...
                -1-7i,-1-5i,-1-1i,-1-3i,-1+7i,-1+5i,-1+1i,-1+3i, ...
                -3-7i,-3-5i,-3-1i,-3-3i,-3+7i,-3+5i,-3+1i,-3+3i, ...
                +7-7i,+7-5i,+7-1i,+7-3i,+7+7i,+7+5i,+7+1i,+7+3i, ...
                +5-7i,+5-5i,+5-1i,+5-3i,+5+7i,+5+5i,+5+1i,+5+3i, ...
                +1-7i,+1-5i,+1-1i,+1-3i,+1+7i,+1+5i,+1+1i,+1+3i, ...
                +3-7i,+3-5i,+3-1i,+3-3i,+3+7i,+3+5i,+3+1i,+3+3i ];
        otherwise
            error('constellation not supported!')
    end

    % normalize symbol energy
    par.symbols = par.symbols/sqrt(mean(abs(par.symbols).^2));

    % set the sampling rate (size of DFT)
    switch par.S
        case 72
            par.N = 128*par.sampling_rate_multiplier;
        case 144
            par.N = 256*par.sampling_rate_multiplier;
        case 300
            par.N = 512*par.sampling_rate_multiplier;
        case 600
            par.N = 1024*par.sampling_rate_multiplier;
        case 900
            par.N = 1536*par.sampling_rate_multiplier;
        case 1200
            par.N = 2048*par.sampling_rate_multiplier;
    end
    
    % occupied/guard subcarriers
    par.submap = [par.N-par.S/2+1:par.N, 2:ceil(par.S/2)+1];
    par.guardmap = setdiff(1:par.N, par.submap);
    
    % sampling rate
    par.samplingrate = 15e3*par.N;
    
    % precompute bit labels
    par.card = length(par.symbols); % cardinality
    par.bps = log2(par.card); % number of bits per symbol
    par.bits = de2bi(0:par.card-1,par.bps,'left-msb'); % symbols-to-bits
            
    % initialize result arrays
    res.BER_uncoded = zeros(length(par.precoder),length(par.SNRdB_list));
    [res.TxAvgPower, res.TxMaxPower] = deal(zeros(length(par.precoder),1));
    [res.MSE, res.EVM] = deal(zeros(length(par.precoder),par.U,par.trials));
    
    psd_xf_emp = zeros(length(par.precoder), par.N);
    psd_yf_emp = zeros(length(par.precoder), par.N);
    
    % -- start simulation

    % save detected symbols for later viewing (if not too many of them)
    if par.trials * par.S <= 1e5
        shat_list = nan(par.U, par.S,par.trials,length(par.precoder));
    end

    % track simulation time
    time_elapsed = 0; tic;

    % trials loop
    for tt = 1:par.trials
        
        % time-domain channel matrix
        Ht = sqrt(0.5/par.T)*(randn(par.U,par.B,par.T) + 1i*randn(par.U,par.B,par.T));
        
        % frequency-domain channel matrix
        if par.T > 1
            Hf = fft(Ht,par.N,3); 
        else
            Hf = nan(par.U,par.B,par.N);
            for n = 1:par.N
                Hf(:,:,n) = Ht;
            end
        end

        % generate iid noise vector 
        nt = sqrt(0.5)*(randn(par.U,par.N)+1i*randn(par.U,par.N)); % time domain
        nf = sqrt(1/par.N)*fft(nt,par.N,2); % frequency domain

        % generate random bits and encode them
        b_data = randi([0 1], par.U, par.bps*par.S); % data bits

        % map bits to subcarriers
        s = zeros(par.U,par.N);
        for u = 1:par.U
            idx = reshape(bi2de(reshape(b_data(u,:),par.bps,par.S)','left-msb')+1,par.S,[]); % index
            s(u,par.submap) = par.symbols(idx); 
        end
         
        % algorithm loop
        for pp = 1:length(par.precoder) 

            % noise-independent precoding
            switch par.precoder{pp}
                case {'MRT', 'MRT_inf'} % MRT precoding
                    zf = zeros(par.B,par.N); % precoded frequency-domain vector
                    [zf(:,par.submap), beta, ~] = MRT(par, s(:,par.submap,:), Hf(:,:,par.submap));
                case {'ZF', 'ZF_inf'} % ZF precoding
                    zf = zeros(par.B,par.N); % precoded frequency-domain vector
                    [zf(:,par.submap), beta, ~] = ZF(par, s(:,par.submap,:), Hf(:,:,par.submap));
            end
            
            for ss = 1:length(par.SNRdB_list) % SNR loop

                N0 = 10.^(-par.SNRdB_list(ss)/10); % noise variance 
                
                % noise-dependent precoding
                switch par.precoder{pp}
                    case {'MRT', 'MRT_inf', 'ZF', 'ZF_inf'}
                        % do nothing
                    case {'WF', 'WF_inf'} % WF precoding
                        zf = zeros(par.B,par.N); % precoded frequency-domain vector
                        [zf(:,par.submap), beta, ~] = WF(par, s(:,par.submap), Hf(:,:,par.submap),N0);
                    case 'SQUID'
                        [zf, beta] = SQUID(par,s,Hf,N0);
                    otherwise
                        error('precoder not supported!')
                end
                
                % transform to time domain
                zt = sqrt(par.N)*ifft(zf, par.N, 2); 
                
                % quantize the phase
                switch par.precoder{pp}
                    case {'MRT', 'ZF', 'WF'} % quantize the phase
                        xt = phase_quantizer(par, zt);
                    otherwise % do nothing
                        xt = zt;
                end
                
                % convert transmit signal to frequency domain
                xf = 1/sqrt(par.N)*fft(xt, par.N, 2);

                % transmit signal over wireless channel
                Hxf = nan(par.U,par.N);
                for n = 1:par.N
                    Hxf(:,n,:) = Hf(:,:,n)*squeeze(xf(:,n,:));
                end
                yf = Hxf + sqrt(N0)*nf; % add noise

                % extract transmitted/received power
                res.TxMaxPower(pp) = max(res.TxMaxPower(pp), max(squeeze(sum(sum(abs(xf).^2,1),2)))/par.S);
                res.TxAvgPower(pp) = res.TxAvgPower(pp) + sum(sum(squeeze(sum(abs(xf).^2))))/par.S/par.trials/length(par.SNRdB_list);

                % estimated symbols (remove guard carriers and compensate for gain loss)
                shat = yf(:,par.submap) ./ sqrt(mean(abs(yf(:,par.submap,:)).^2,2) - N0);
                
                % symbol detection  
                b_detected = nan(size(b_data));
                for u = 1:par.U
                    [~,idxhat] = min(abs(reshape(shat(u,:,:),[],1)*ones(1,length(par.symbols))-ones(par.S,1)*par.symbols).^2,[],2);
                    b_detected(u,:) = reshape(par.bits(idxhat,:)',1,[]);
                end
                
                % compute bit error rate
                res.BER_uncoded(pp,ss) = res.BER_uncoded(pp,ss) + sum(sum(b_data~=b_detected))/par.U/par.bps/par.S/par.trials;
                
                % compute average MSE and EVM per UE
                res.MSE(pp,:,tt) = mean(abs(beta*Hxf(:,par.submap) - s(:,par.submap)).^2,2);
                res.EVM(pp,:,tt) = 100*sqrt(res.MSE(pp,:,tt).' ./ mean(abs(s(:,par.submap)).^2,2));
                
            end % end of SNR loop
            
            % save data for later viewing
            if par.trials * par.S <= 1e5
                shat_list(:,:,tt,pp) = reshape(shat,par.U,[]);
            end
            
            % power spectral density (averaged over antennas and trials)
            psd_xf_emp(pp,:) = psd_xf_emp(pp,:) + mean(abs(xf).^2,1)/par.trials;
            psd_yf_emp(pp,:) = psd_yf_emp(pp,:) + mean(abs(Hxf).^2,1)/par.trials;

        end % end of algorithm loop

         % keep track of simulation time
        if toc>10
            
            time=toc;
            time_elapsed = time_elapsed + time;
            fprintf('\t estimated remaining simulation time: %3.0f min. \n',time_elapsed*(par.trials/tt-1)/60);
            tic;
            
        end


    end % end of channels loop

    fprintf('\t numerical simulation finished after %.2f seconds. \n', time_elapsed);

    %----------------------------------------------------------------------
    % -- end of simulation
    %----------------------------------------------------------------------

    if par.plot
        
        % number of rows/columns in const/psd plots
        if length(par.precoder)==1
            nd = 1;
        elseif length(par.precoder)<=6
            nd = 2;
        elseif length(par.precoder)<=9
            nd = 3;
        elseif length(par.precoder)<=12
            nd = 4;
        end
        md = ceil(length(par.precoder)/nd);
        
        % marker style and color
        marker_style = {'o-','s--','v-.','+:','<-','>--','x-.','^:','*-','d--','h-.','p:'};
        marker_color = [...
            0.0000    0.4470    0.7410;...
            0.8500    0.3250    0.0980;...
            0.9290    0.6940    0.1250;...
            0.4940    0.1840    0.5560;...
            0.4660    0.6740    0.1880;...
            0.3010    0.7450    0.9330;...
            0.6350    0.0780    0.1840;...
            0.7500    0.7500    0.0000;...
            0.7500    0.0000    0.7500;...
            0.0000    0.5000    0.0000;...
            0.0000    0.0000    1.0000;...
            1.0000    0.0000    0.0000];
        
        % legends
        precoder_legend = par.precoder;
        for pp = 1:length(par.precoder)
            if strcmpi(precoder_legend{pp}, 'MRT')
                precoder_legend{pp} = [num2str(par.P),'-phase-bit MRT'];
            elseif strcmpi(precoder_legend{pp}, 'MRT_inf')
                precoder_legend{pp} = 'Infinite-resolution MRT';
            elseif strcmpi(precoder_legend{pp}, 'ZF')
                precoder_legend{pp} = [num2str(par.P),'-phase-bit ZF'];
            elseif strcmpi(precoder_legend{pp}, 'ZF_inf')
                precoder_legend{pp} = 'Infinite-resolution ZF';
            elseif strcmpi(precoder_legend{pp}, 'WF')
                precoder_legend{pp} = [num2str(par.P),'-phase-bit WF'];
            elseif strcmpi(precoder_legend{pp}, 'WF_inf')
                precoder_legend{pp} = 'Infinite-resolution WF';
            elseif strcmpi(precoder_legend{pp}, 'SQUID')
                precoder_legend{pp} = [num2str(par.P),'-phase-bit SQUID'];
            end
        end
        
        % limits
        ylim_min = -50; 
        ylim_max = ceil(10*log10(max([max(abs(psd_xf_emp(pp,:))), max(abs(psd_yf_emp(pp,:)))])))+10;
        
        % transmitted spectrum
        fig_txspec = figure; set(fig_txspec,'name','Tx Spectrum','numbertitle','off');
        for pp = 1:length(par.precoder)
            subplot(md,nd,pp); hold all;
            plot(15e-3*(-par.N/2:par.N/2-1),10*log10(eps+fftshift(psd_xf_emp(pp,:)+eps)),'-x','color',marker_color(pp,:));
            xlim(15e-3*[-ceil(par.N/2), ceil(par.N/2)-1]); ylim([ylim_min, ylim_max]);
            xlabel('Frequency [MHz]','fontsize',14); ylabel('PSD [dB]','fontsize',14); box on; grid on;
            title(precoder_legend{pp},'fontsize',12);
        end

        % received spectrum
        fig_rxspec = figure; set(fig_rxspec,'name','Rx Spectrum','numbertitle','off');
        for pp = 1:length(par.precoder)
            subplot(md,nd,pp); hold all;
            plot(15e-3*(-par.N/2:par.N/2-1),10*log10(eps+fftshift(psd_yf_emp(pp,:)+eps)),'-x','color',marker_color(pp,:));
            xlim(15e-3*[-ceil(par.N/2), ceil(par.N/2)-1]); ylim([ylim_min, ylim_max]);
            xlabel('Frequency [MHz]','fontsize',14); ylabel('PSD [dB]','fontsize',14); box on; grid on;
            title(precoder_legend{pp},'fontsize',12);
        end
        
        % plot constellation
        if par.trials * par.S <= 1e5
            
            fig_const = figure; set(fig_const,'name','Const.','numbertitle','off');
            for pp = 1:length(par.precoder)
                subplot(md,nd,pp); hold all;
                plot(reshape(shat_list(:,:,:,pp),1,[]),'*', 'color', marker_color(pp,:),'markersize',7);
                plot(par.symbols, 'ko','MarkerSize',7);
                axis(max(reshape(abs(shat_list(:,:,:,pp)),1,[]))*[-1 1 -1 1]); 
                axis square; box on;
                title(precoder_legend{pp},'fontsize',12);
                xlabel(['P_{avg}= ',num2str(pow2db(res.TxAvgPower(pp)),'%0.2f'),' dB  and  P_{max}= ',num2str(pow2db(res.TxMaxPower(pp)),'%0.2f'),' dB'],'fontsize',12);
            end

        end

        % plot uncoded BER
        fig_uncodedber = figure; set(fig_uncodedber,'name','Uncoded BER','numbertitle','off');
        for pp=1:length(par.precoder) % simulated BER
            semilogy(par.SNRdB_list,res.BER_uncoded(pp,:),marker_style{pp},'color',marker_color(pp,:),'LineWidth',2); hold on;
        end
        grid on; box on;
        xlabel('SNR [dB]','FontSize',12)
        ylabel('uncoded BER','FontSize',12);
        if length(par.SNRdB_list) > 1
            axis([min(par.SNRdB_list) max(par.SNRdB_list) 1e-4 1]);
        end
        legend(precoder_legend,'FontSize',12,'location','southwest')
        set(gca,'FontSize',12);
        
    end
        
    % save final results
    if par.save
        save(par.simName,'par','res');
    end
    
end

% -- precoders

function [xf, beta, Pf] = MRT(par, s, Hf)
% -------------------------------------------------------------------------
% maximal-ratio transmission (MRT)
% -------------------------------------------------------------------------
%   -- inputs:
%       - par: struct with system parameters
%       - s: UxSxMAX complex-valued symbol vector
%       - Hf: UxBxS complex-valued frequency-domain channel matrix
%   -- outputs: 
%       - x: BxSxMAX complex-valued frequency-domain precoded vector
%       - beta: real-valued scalar precoding factor
%       - Pf: UxBxS complex-valued frequency-domain precoding matrix
% -------------------------------------------------------------------------
% 2018 (c) sven.jacobsson@ericsson.com and studer@cornell.edu
% -------------------------------------------------------------------------

    % initialize vectors
    xf = zeros(par.B,par.S); 
    Pf = nan(par.B,par.U,par.S);
    beta = 0;

    % precoding
    for k = 1:par.S
        Pf(:,:,k) = 1/par.B*Hf(:,:,k)'; % precoding matrix
        xf(:,k,:) = Pf(:,:,k)*reshape(s(:,k,:),par.U,[]); % precoded vector
        beta = beta + trace(Pf(:,:,k)*Pf(:,:,k)')/par.S; % precoding factor (squared) 
    end
    beta = sqrt(beta); % precoding factor
    
    % rescale output
    xf = xf/beta;
    Pf = Pf/beta; 

                
end

function [xf, beta, Pf] = ZF(par, s, Hf)
% -------------------------------------------------------------------------
% zero-forcing (ZF)
% -------------------------------------------------------------------------
%   -- inputs:
%       - par: struct with system parameters
%       - s: UxSxMAX complex-valued symbol vector
%       - Hf: UxBxS complex-valued frequency-domain channel matrix
%   -- outputs: 
%       - x: BxSxMAX complex-valued frequency-domain precoded vector
%       - beta: real-valued scalar precoding factor
%       - Pf: UxBxS complex-valued frequency-domain precoding matrix
% -------------------------------------------------------------------------
% 2018 (c) sven.jacobsson@ericsson.com and studer@cornell.edu
% -------------------------------------------------------------------------
    
    % initialize vectors
    xf = zeros(par.B,par.S); 
    Pf = nan(par.B,par.U,par.S);
    beta = 0;
    
    % precoding
    for k = 1:par.S
        Pf(:,:,k) = Hf(:,:,k)'/(Hf(:,:,k)*Hf(:,:,k)'); % precoding matrix
        xf(:,k,:) = Pf(:,:,k)*reshape(s(:,k,:),par.U,[]); % precoded vector
        beta = beta + trace(Pf(:,:,k)*Pf(:,:,k)')/par.S; % precoding factor (squared) 
    end
    beta = sqrt(beta); % precoding factor
    
    % rescale output
    xf = xf/beta;
    Pf = Pf/beta; 
    
end

function [xf, beta, Pf] = WF(par, s, Hf, N0)
% -------------------------------------------------------------------------
% Wiener-filter (WF)
% -------------------------------------------------------------------------
%   -- inputs:
%       - par: struct with system parameters
%       - s: UxSxMAX complex-valued symbol vector
%       - Hf: UxBxS complex-valued frequency-domain channel matrix
%       - N0: noise power
%   -- outputs: 
%       - x: BxSxMAX complex-valued frequency-domain precoded vector
%       - beta: real-valued scalar precoding factor
%       - Pf: UxBxS complex-valued frequency-domain precoding matrix
% -------------------------------------------------------------------------
% 2018 (c) sven.jacobsson@ericsson.com and studer@cornell.edu
% -------------------------------------------------------------------------
    
    % initialize vectors
    xf = zeros(par.B,par.S); 
    Pf = nan(par.B,par.U,par.S);
    beta = 0;
    
    % precoding
    for k = 1:par.S
        Pf(:,:,k) = Hf(:,:,k)'/(Hf(:,:,k)*Hf(:,:,k)' + N0*par.U*eye(par.U)); % precoding matrix
        xf(:,k,:) = Pf(:,:,k)*squeeze(s(:,k,:)); % precoded vector
        beta = beta + trace(Pf(:,:,k)*Pf(:,:,k)')/par.S; % precoding factor (squared) 
    end
    beta = sqrt(beta); % precoding factor
    
    % rescale output
    xf = xf/beta;
    Pf = Pf/beta; 
    
end

function [xf, beta] = SQUID(par,s,Hf,N0)
% -------------------------------------------------------------------------
% squared infinity-norm Douglas-Rachford splitting (SQUID)
% -------------------------------------------------------------------------
%   -- inputs:
%       - par: struct with system parameters
%       - s: UxS complex-valued symbol vector
%       - Hf: UxBxS complex-valued frequency-domain channel matrix
%       - N0: noise power
%   -- outputs: 
%       - x: BxS complex-valued frequency-domain precoded vector
%       - beta: real-valued scalar precoding factor
% -------------------------------------------------------------------------
% 2018 (c) sven.jacobsson@ericsson.com and studer@cornell.edu
% -------------------------------------------------------------------------

    % tuning parameters (should be optimized for best performance!)
    iter = 20; % number of iterations
    rho = 1; % relxation parameter /
    gain = 1; % set to 1 for large problems; small values for small, ill-conditioned problems
    
    % preprocessing
    Qf = nan(par.B,par.U,par.N);
    df = nan(par.B,par.N);
    for k = 1:par.N 
        Qf(:,:,k) = Hf(:,:,k)'/(Hf(:,:,k)*Hf(:,:,k)' + (0.5/gain)*eye(par.U));
        zMF = Hf(:,:,k)'*s(:,k);
        df(:,k) = (2*gain)*(zMF - Qf(:,:,k)*(Hf(:,:,k)*zMF));
    end
    
    % initialization
    Af = zeros(par.B,par.N); 
    Bf = zeros(par.B,par.N);
    Cf = zeros(par.B,par.N);
    
    for i = 1:iter % SQUID loop
        
        % first step: frequency-domain MMSE precoding
        for k = 1:par.N 
            zfk = 2*Bf(:,k) - Cf(:,k);
            if ismember(k, par.submap) % minimize MSE for occupied subcarriers
                Af(:,k) = df(:,k) + zfk - Qf(:,:,k)*(Hf(:,:,k)*zfk);
            else % do nothing for guard subcarriers 
                Af(:,k) = zfk;
            end
        end
        gamma = 2*par.U*par.B*par.N*N0;
        
        % convert to time-domain and vectorize matrix
        wtv = reshape(sqrt(par.N)*ifft(Cf + Af - Bf,par.N,2), par.B*par.N, 1);
        wtvR = [real(wtv); imag(wtv)];
        
        % second step: time-domain constant-envelope constraint
        switch par.P 
            case 1 % 1-phase-bit output
                btv = 1i*prox_infinity_norm_squared(imag(wtv), gamma/2);
                Bt = reshape(btv, par.B, par.N);
            case 2 % 2-phase-bit output
                btv = prox_infinity_norm_squared(wtvR, gamma); 
                Bt = reshape(btv(1:par.B*par.N) + 1i*btv(par.B*par.N + (1:par.B*par.N)), par.B, par.N);
            otherwise % 3-phase-bit or constant-envelope output
                btv = prox_infinity_norm_squared(wtv, gamma/2);  
                Bt = reshape(btv, par.B, par.N);  
        end
        Bf = fft(Bt,par.N,2)/sqrt(par.N);
        
        % third step: update with under/over-relaxation
        Cf = Cf + rho*(Af - Bf); 
        
    end
    
    % quantize the phase 
    xt = phase_quantizer(par, Bt);
    
    % convert to frequency-domain
    xf = fft(xt,par.N,2)/sqrt(par.N);   
    
    % compute precoding factor
    beta = 0;
    for k = 1:par.S
        Hx = Hf(:,:,par.submap(k))*xf(:,par.submap(k));
        beta = beta + real(Hx'*s(:,par.submap(k)))/(norm(Hx,2)^2+par.U*N0)/par.S;
    end

end

% -- auxilliary functions

% phase quantization 
function xt = phase_quantizer(par, zt)

    switch par.P 
        case 1 % 1-phase-bit output
            xt = sqrt(par.S/(par.B*par.N))*exp(1i*pi * (floor(1/pi*angle(zt)) + 1/2));
        case 2 % 2-phase-bit output
            xt = sqrt(par.S/(par.B*par.N))*exp(1i*2*pi/4 * (floor(4/(2*pi)*angle(zt)) + 1/2));
        case 3 % 3-phase-bit output
            xt = sqrt(par.S/(par.B*par.N))*exp(1i*2*pi/8 * (floor(8/(2*pi)*angle(zt)) + 1/2));    
        case inf % constant-envelope output
            xt = sqrt(par.S/(par.B*par.N))*exp(1i*angle(zt));
    end
    
end

% proximal operator for the squared infinity norm 
function wp = prox_infinity_norm_squared(w,lambda)

    wabs = abs(w);
    ws = (cumsum(sort(wabs,'descend')))./(2*lambda+(1:length(w))');
    
    alphaopt = max(ws);
    if alphaopt>0 
      wp = min(wabs,alphaopt).*sign(w); % truncation step
    else
      wp = zeros(size(w)); % if lambda is big, then solution is zero
    end   
    
end

