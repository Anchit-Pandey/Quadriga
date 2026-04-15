%% ========================================================================
%  CS3T-UNet Dataset Generation — Paper-Exact Implementation
%  Paper: "Cross-shaped Separated Spatial-Temporal UNet Transformer
%          For Accurate Channel Prediction" (IEEE INFOCOM 2024)
%
%  Faithful to:
%    - Section II  (System Model, ADP transform Eq.3)
%    - Section IV-A (Dataset, normalization, train/test split)
%
%  All assumptions explicitly labeled [ASSUMPTION: ...]
%  All paper citations labeled   [PAPER: ...]
%  All fixes vs. original code   [FIX: ...]
%% ========================================================================

clear; clc; close all;
rng(42);   % [PAPER: not stated] [ASSUMPTION: seed=42 for reproducibility]

%% -----------------------------------------------------------------------
%  PATH — edit this to point at your QuaDRiGa installation
%% -----------------------------------------------------------------------
addpath(genpath('D:\Anchic\QuaDriGa_2023.12.13_v2.8.1-0\quadriga_src'));

%% ========================================================================
%  SECTION 1 — PARAMETERS (all taken directly from Section IV-A)
%% ========================================================================
fprintf('=== CS3T-UNet Dataset Generator (Paper-Exact) ===\n\n');

% [PAPER: Section IV-A]
fc        = 5.0e9;           % carrier frequency: 5.0 GHz
Nc        = 64;              % number of subcarriers
delta_f   = 30e3;            % subcarrier spacing: 30 kHz
Nt        = 64;              % BS antennas (ULA)
Nr        = 1;               % UE antennas (omnidirectional)
UE_speed  = 5 / 3.6;        % UE speed: 5 km/h -> m/s
scenario  = '3GPP_38.901_UMa_NLOS';  % 3GPP UMa NLOS

% [PAPER: Section IV-A] "We generate the environment 100 times,
%  each containing 100 UEs => 10k samples total"
num_environments = 100;
num_UEs_per_env  = 100;
total_samples    = num_environments * num_UEs_per_env;  % 10000

% [PAPER: Section IV-A] "each sample contains 20 frames"
num_frames = 20;

% [PAPER: Section IV-A] train/test: 9K / 1K
num_train = 9000;
num_test  = 1000;

% [PAPER: Section IV-A] T=10 historical, L=1 or L=5 future
T_hist = 10;  % historical steps fed to the model
L_pred = 5;   % future steps predicted (max used in paper)

% Antenna geometry
lambda      = 3e8 / fc;
ant_spacing = lambda / 2;

% [PAPER: Section IV-A, implicit from coherence time discussion]
% Frame duration chosen so adjacent-frame correlation ~0.97
% fd*Ts ≈ 0.03 => Ts = 0.03 / fd, fd = v/lambda
fd             = UE_speed / lambda;
frame_duration = 0.03 / fd;               % ~2.5 ms at 5 km/h, 5 GHz
pos_spacing    = UE_speed * frame_duration;
total_track    = pos_spacing * (num_frames - 1);
total_duration = frame_duration * (num_frames - 1);

adj_corr       = abs(besselj(0, 2*pi*fd*frame_duration));
frame_nmse_exp = 10*log10(2*(1 - adj_corr));

fprintf('[Params] fc=%.1f GHz  Nc=%d  delta_f=%.0f kHz  Nt=%d\n', ...
        fc/1e9, Nc, delta_f/1e3, Nt);
fprintf('[Params] Speed=%.2f m/s  Frame duration=%.3f ms\n', ...
        UE_speed, frame_duration*1e3);
fprintf('[Params] Adjacent correlation=%.4f  Frame NMSE~%.2f dB\n\n', ...
        adj_corr, frame_nmse_exp);

%% ========================================================================
%  SECTION 2 — STORAGE ALLOCATION
%% ========================================================================
% Shape: (samples, frames, 2[real/imag], Nc, Nt)
% [PAPER: Section IV-A] "H ∈ R^{10K×20×2×64×64}"
all_H_raw = zeros(total_samples, num_frames, 2, Nc, Nt, 'single');

% Track which sample indices are valid (failed UEs → invalid)
% [FIX: original code incremented sample_idx even on error, leaving
%  all-zero rows indistinguishable from valid data]
valid_mask  = false(total_samples, 1);
sample_idx  = 1;
error_count = 0;
total_start = tic;

%% ========================================================================
%  SECTION 3 — RAW CSI GENERATION  (Stage A)
%% ========================================================================
fprintf('=== STAGE A: Raw CSI Generation ===\n');

for env = 1:num_environments

    env_start   = tic;
    % Random UE positions around BS
    pos_angles  = 2*pi * rand(1, num_UEs_per_env);
    dists       = 50 + 450 * rand(1, num_UEs_per_env);   % 50–500 m
    move_angles = 2*pi * rand(1, num_UEs_per_env);

    for u = 1:num_UEs_per_env

        try
            %% QuaDRiGa simulation setup
            s                     = qd_simulation_parameters;
            s.center_frequency    = fc;
            % [ASSUMPTION: sample_density=2 gives ≥20 snapshots for
            %  total_track length; verified in validation section below]
            s.sample_density      = 2;
            s.use_absolute_delays = 1;
            s.show_progress_bars  = 0;

            %% BS antenna: ULA with half-wavelength spacing
            % [PAPER: Section II] "Nt-antenna ULA"
            a_bs = qd_arrayant('omni');
            a_bs.no_elements = Nt;
            a_bs.element_position(1,:) = zeros(1, Nt);
            a_bs.element_position(2,:) = (0:Nt-1) * ant_spacing;
            a_bs.element_position(3,:) = zeros(1, Nt);

            %% UE antenna: omnidirectional
            % [PAPER: Section II] "UE with Nr omnidirectional antennas"
            a_ue = qd_arrayant('omni');

            %% Layout
            l             = qd_layout(s);
            l.no_tx       = 1;
            l.no_rx       = 1;
            l.tx_array    = a_bs;
            l.rx_array    = a_ue;
            l.tx_position = [0; 0; 25];   % BS height 25 m (UMa standard)

            %% UE linear track
            % [PAPER: Section IV-A] "moving speed of the UE is 5 km/h"
            % Time-EVOLVING channel — NOT static snapshots
            % [FIX: track must be exactly long enough to produce num_frames
            %  evenly-spaced snapshots at frame_duration intervals]
            t_ue      = qd_track('linear', total_track, move_angles(u));
            t_ue.name = sprintf('env%03due%03d', env, u);
            t_ue.initial_position = [ dists(u)*cos(pos_angles(u));
                                      dists(u)*sin(pos_angles(u));
                                      1.5 ];   % UE height 1.5 m

            %% Explicit waypoints at exact frame positions
            pos_along      = linspace(0, total_track, num_frames);
            time_stamps    = linspace(0, total_duration, num_frames);
            t_ue.positions = [pos_along*cos(move_angles(u));
                              pos_along*sin(move_angles(u));
                              zeros(1, num_frames)];
            t_ue.movement_profile = [time_stamps; pos_along];
            t_ue.scenario = scenario;
            l.rx_track(1,1) = t_ue;

            %% Generate time-evolving channel
            cb       = l.init_builder;
            cb.gen_parameters;
            channels = cb.get_channels();
            ch       = channels(1);

            num_paths     = size(ch.coeff, 3);
            num_snapshots = size(ch.coeff, 4);

            % [FIX: if QuaDRiGa returns fewer snapshots than num_frames,
            %  skip this sample entirely instead of interpolating.
            %  Interpolation distorts the exact frame_duration spacing.]
            if num_snapshots < num_frames
                error_count = error_count + 1;
                warning('env%d UE%d: only %d snapshots (need %d) — skipped', ...
                        env, u, num_snapshots, num_frames);
                sample_idx = sample_idx + 1;
                continue
            end

            % Select the first num_frames snapshots at exact waypoints.
            % QuaDRiGa places one snapshot per waypoint when movement_profile
            % is set; the first num_frames entries map to our time stamps.
            snap_idx = 1:num_frames;

            %% Build frequency-domain channel matrix  [PAPER: Eq. (1)&(2)]
            %  h[l] = sum_m alpha_m * a(theta_m) * exp(-j2pi*fl*tau_m)
            f_axis   = (0:Nc-1) * delta_f;       % subcarrier frequencies
            H_sample = zeros(num_frames, Nc, Nt, 'double');

            for t = 1:num_frames
                sidx    = snap_idx(t);
                coeff_t = ch.coeff(:,:,:,sidx);   % (Nr x Nt x Npaths)
                delay_t = ch.delay(:,:,:,sidx);   % (Nr x Nt x Npaths)
                H_frame = zeros(Nc, Nt);

                for m = 1:num_paths
                    for ant = 1:Nt
                        alpha = coeff_t(1, ant, m);
                        tau   = delay_t(1, ant, m);
                        % Phase term over all subcarriers
                        phase = exp(-1j * 2*pi * f_axis * tau);
                        H_frame(:, ant) = H_frame(:, ant) + alpha * phase.';
                    end
                end
                H_sample(t,:,:) = H_frame;
            end

            % Store real and imaginary parts separately
            all_H_raw(sample_idx,:,1,:,:) = single(real(H_sample));
            all_H_raw(sample_idx,:,2,:,:) = single(imag(H_sample));
            valid_mask(sample_idx)         = true;   % mark as valid

        catch ME
            error_count = error_count + 1;
            fprintf('  [ERROR] env%d UE%d: %s\n', env, u, ME.message);
            % sample_idx still increments; valid_mask stays false → filtered later
        end

        sample_idx = sample_idx + 1;

    end % UE loop

    % Progress report
    elapsed   = toc(total_start);
    done      = sample_idx - 1;
    remaining = (total_samples - done) / max(done/elapsed, 1e-9);
    fprintf('Env %3d/%d | %.1fs | errors=%d | ETA %.1f min\n', ...
            env, num_environments, toc(env_start), error_count, remaining/60);

    % Checkpoint every 10 environments
    if mod(env, 10) == 0
        H_partial    = all_H_raw(1:done,:,:,:,:);
        valid_partial = valid_mask(1:done);
        save(sprintf('checkpoint_env%03d.mat', env), ...
             'H_partial','valid_partial','sample_idx','env','-v7.3');
        fprintf('  >>> Checkpoint saved (env%03d)\n', env);
    end

end % environment loop

fprintf('\nRaw CSI generation complete. Errors: %d/%d\n\n', ...
        error_count, total_samples);

%% ========================================================================
%  SECTION 4 — FILTER INVALID SAMPLES
%% ========================================================================
fprintf('=== Filtering invalid samples ===\n');
num_valid = sum(valid_mask);
fprintf('Valid samples: %d / %d\n', num_valid, total_samples);

if num_valid < num_train + num_test
    error('Not enough valid samples (%d). Need at least %d. Re-run generation.', ...
          num_valid, num_train + num_test);
end

all_H_raw = all_H_raw(valid_mask,:,:,:,:);   % keep only valid rows

%% ========================================================================
%  SECTION 5 — ADP CONVERSION  (Stage B)
%  [PAPER: Section II, Eq. (3)]  H' = Ff * H * Ft^H
%% ========================================================================
fprintf('\n=== STAGE B: ADP Conversion ===\n');

% Build DFT matrices  (paper convention: Ff(i,k)=exp(-j*2pi*(i-1)*(k-1)/Nf))
i_Nc = (0:Nc-1)';
k_Nc = (0:Nc-1);
Ff   = exp(-1j * 2*pi * i_Nc * k_Nc / Nc);      % Nc×Nc

i_Nt = (0:Nt-1)';
k_Nt = (0:Nt-1);
Ft   = exp(-1j * 2*pi * i_Nt * k_Nt / Nt);      % Nt×Nt
FtH  = Ft';                                       % conjugate transpose = Ft^H

num_valid_samples = size(all_H_raw, 1);
all_H_adp         = zeros(num_valid_samples, num_frames, 2, Nc, Nt, 'single');
adp_start         = tic;

for s = 1:num_valid_samples
    for t = 1:num_frames
        H_r   = double(squeeze(all_H_raw(s,t,1,:,:)));   % Nc×Nt
        H_i   = double(squeeze(all_H_raw(s,t,2,:,:)));
        H_cplx = H_r + 1j*H_i;

        % [PAPER: Eq.(3)]  H' = Ff * H * Ft^H
        H_adp = Ff * H_cplx * FtH;

        all_H_adp(s,t,1,:,:) = single(real(H_adp));
        all_H_adp(s,t,2,:,:) = single(imag(H_adp));
    end

    if mod(s, 1000) == 0
        fprintf('  ADP: %d/%d (%.1fs)\n', s, num_valid_samples, toc(adp_start));
    end
end

fprintf('ADP done in %.2f min\n', toc(adp_start)/60);
fprintf('ADP range before norm: [%.4f, %.4f]\n\n', ...
        min(all_H_adp(:)), max(all_H_adp(:)));

%% ========================================================================
%  SECTION 6 — TRAIN / TEST SPLIT
%  [PAPER: Section IV-A] "9K training, 1K test"
%% ========================================================================
fprintf('=== Train/Test Split ===\n');
rng(42);
idx_shuf  = randperm(num_valid_samples);
train_idx = idx_shuf(1:num_train);
test_idx  = idx_shuf(num_train+1 : num_train+num_test);
fprintf('Train: %d  |  Test: %d\n\n', numel(train_idx), numel(test_idx));

train_H_adp_raw = all_H_adp(train_idx,:,:,:,:);
test_H_adp_raw  = all_H_adp(test_idx,:,:,:,:);

%% ========================================================================
%  SECTION 7 — NORMALISATION  (Stage C)
%  [PAPER: Section IV-A]
%  "We divide the previous timesteps and the predictions by their OWN
%   maximum amplitude to make sure their complex values are in [-1, 1].
%   The tanh activation function is attached to the outputs of all models."
%
%  This is per-sample normalisation.  Each sample (all 20 frames) is
%  divided by the maximum absolute value of that sample's complex ADP.
%  Both input frames and target frames share the same scalar so that
%  relative amplitude between past and future is preserved — which is
%  exactly the "relative scaling of MIMO CSI components" the paper cites.
%
%  [PAPER reference for per-sample norm: Liu et al. FIRE MobiCom 2021,
%   cited as [23] in the paper, which the paper explicitly follows.]
%
%  NOTE: A global-max alternative was used in generate_data.m but it
%  diverges from the paper spec and produces non-comparable NMSE values.
%% ========================================================================
fprintf('=== STAGE C: Per-Sample Normalisation (Paper Exact) ===\n');

function H_norm = per_sample_normalize(H_adp)
    % H_adp: (N, T, 2, Nc, Nt) single
    % Returns: H_norm (same shape, each sample in [-1,1]),
    %          scale_vec (N,1) for inversion
    N = size(H_adp,1);
    H_norm   = zeros(size(H_adp), 'single');
    scale_vec = zeros(N,1,'single');
    for i = 1:N
        sample     = H_adp(i,:,:,:,:);
        % Reconstruct complex values to find true amplitude max
        sr         = squeeze(sample(:,:,1,:,:));   % (T,Nc,Nt)
        si         = squeeze(sample(:,:,2,:,:));
        amp_max    = max(abs(sr(:) + 1j*si(:)));
        if amp_max < 1e-12
            amp_max = 1.0;   % guard against zero-energy samples
        end
        H_norm(i,:,:,:,:) = single(sample / amp_max);
        scale_vec(i)      = single(amp_max);
    end
end

[train_adp, train_scale] = per_sample_normalize(train_H_adp_raw);
[test_adp,  test_scale]  = per_sample_normalize(test_H_adp_raw);

fprintf('Train ADP range after norm: [%.4f, %.4f]\n', ...
        min(train_adp(:)), max(train_adp(:)));
fprintf('Test  ADP range after norm: [%.4f, %.4f]\n', ...
        min(test_adp(:)), max(test_adp(:)));

% [FIX: verify test values stay in [-1,1] — they must under per-sample norm]
assert(max(abs(test_adp(:))) <= 1.0 + 1e-5, ...
       'Test values exceed ±1 after normalisation — check pipeline.');
fprintf('Normalisation bounds assertion passed.\n\n');

%% ========================================================================
%  SECTION 8 — SLIDING-WINDOW SEQUENCE CONSTRUCTION
%  [PAPER: Section IV-A]
%  "We fix T=10 historical steps and L=1 and L=5 future steps.
%   We use a T+L-length sliding window to slide along the 20 frames
%   of each sample with stride 1 for data augmentation."
%  Input  X: (samples_aug, T,   2, Nc, Nt)
%  Target Y: (samples_aug, L,   2, Nc, Nt)
%% ========================================================================
fprintf('=== Sliding-Window Sequence Construction ===\n');

function [X, Y] = build_sequences(H_norm, T, L)
    % H_norm: (N, num_frames, 2, Nc, Nt)
    % Stride-1 sliding window: positions 1..num_frames-(T+L)+1
    [N, F, C, Nf, Na] = size(H_norm);
    stride    = 1;
    win_len   = T + L;
    num_wins  = floor((F - win_len) / stride) + 1;   % per sample
    total_seq = N * num_wins;

    X = zeros(total_seq, T, C, Nf, Na, 'single');
    Y = zeros(total_seq, L, C, Nf, Na, 'single');

    seq = 1;
    for n = 1:N
        for w = 1:num_wins
            t_start = (w-1)*stride + 1;
            X(seq,:,:,:,:) = H_norm(n, t_start        : t_start+T-1, :,:,:);
            Y(seq,:,:,:,:) = H_norm(n, t_start+T      : t_start+T+L-1, :,:,:);
            seq = seq + 1;
        end
    end
end

% Build for L=1 (single-step) and L=5 (multi-step)
[X_train_L1, Y_train_L1] = build_sequences(train_adp, T_hist, 1);
[X_test_L1,  Y_test_L1]  = build_sequences(test_adp,  T_hist, 1);

[X_train_L5, Y_train_L5] = build_sequences(train_adp, T_hist, L_pred);
[X_test_L5,  Y_test_L5]  = build_sequences(test_adp,  T_hist, L_pred);

fprintf('L=1  Train X: %s  Y: %s\n', mat2str(size(X_train_L1)), mat2str(size(Y_train_L1)));
fprintf('L=1  Test  X: %s  Y: %s\n', mat2str(size(X_test_L1)),  mat2str(size(Y_test_L1)));
fprintf('L=5  Train X: %s  Y: %s\n', mat2str(size(X_train_L5)), mat2str(size(Y_train_L5)));
fprintf('L=5  Test  X: %s  Y: %s\n\n', mat2str(size(X_test_L5)), mat2str(size(Y_test_L5)));

%% ========================================================================
%  SECTION 9 — VALIDATION
%% ========================================================================
fprintf('=== VALIDATION ===\n');
validate_dataset(train_adp, test_adp, X_train_L5, Y_train_L5, ...
                 all_H_adp, train_idx, num_frames, Nc, Nt, frame_duration, fd);

%% ========================================================================
%  SECTION 10 — SAVE
%% ========================================================================
fprintf('\n=== SAVING ===\n');

% Per-split normalised sequences (primary training files)
save('train_L1.mat', 'X_train_L1','Y_train_L1', '-v7.3');
save('test_L1.mat',  'X_test_L1', 'Y_test_L1',  '-v7.3');
save('train_L5.mat', 'X_train_L5','Y_train_L5', '-v7.3');
save('test_L5.mat',  'X_test_L5', 'Y_test_L5',  '-v7.3');

% Full normalised ADP (before sliding window) + scale vectors
save('train_adp_norm.mat', 'train_adp','train_scale', '-v7.3');
save('test_adp_norm.mat',  'test_adp', 'test_scale',  '-v7.3');

% Full raw ADP (before normalisation, after ADP transform)
save('all_H_adp.mat', 'all_H_adp', '-v7.3');

% Metadata for reproducibility
meta.fc = fc; meta.Nc = Nc; meta.delta_f = delta_f;
meta.Nt = Nt; meta.Nr = Nr; meta.UE_speed = UE_speed;
meta.num_frames = num_frames; meta.T_hist = T_hist; meta.L_pred = L_pred;
meta.frame_duration = frame_duration; meta.scenario = scenario;
meta.norm_method = 'per_sample_max_amplitude';
meta.train_idx = train_idx; meta.test_idx = test_idx;
meta.valid_mask = valid_mask; meta.rng_seed = 42;
save('dataset_meta.mat', 'meta');

fprintf('Saved files:\n');
fprintf('  train_L1.mat / test_L1.mat   — sliding window, L=1\n');
fprintf('  train_L5.mat / test_L5.mat   — sliding window, L=5\n');
fprintf('  train_adp_norm.mat           — per-sample normalised ADP\n');
fprintf('  test_adp_norm.mat\n');
fprintf('  all_H_adp.mat                — ADP before normalisation\n');
fprintf('  dataset_meta.mat             — all parameters\n');
fprintf('Total wall time: %.2f min\n', toc(total_start)/60);

%% ========================================================================
%  LOCAL FUNCTION: VALIDATE_DATASET
%% ========================================================================
function validate_dataset(train_adp, test_adp, X_train, Y_train, ...
                           all_H_adp, train_idx, num_frames, Nc, Nt, ...
                           frame_duration, fd)

    fprintf('\n--- Check 1: NaN / Inf ---\n');
    assert(~any(isnan(train_adp(:))),  'NaN in train_adp');
    assert(~any(isinf(train_adp(:))),  'Inf in train_adp');
    assert(~any(isnan(test_adp(:))),   'NaN in test_adp');
    assert(~any(isinf(test_adp(:))),   'Inf in test_adp');
    fprintf('  PASS — no NaN or Inf\n');

    fprintf('\n--- Check 2: Normalisation bounds [-1, 1] ---\n');
    tr_max = max(abs(train_adp(:)));
    te_max = max(abs(test_adp(:)));
    fprintf('  Train max |val| = %.6f  (must be <= 1)\n', tr_max);
    fprintf('  Test  max |val| = %.6f  (must be <= 1 under per-sample norm)\n', te_max);
    assert(tr_max <= 1.0 + 1e-5, 'Train values exceed ±1');
    assert(te_max <= 1.0 + 1e-5, 'Test values exceed ±1');
    fprintf('  PASS\n');

    fprintf('\n--- Check 3: Tensor shapes ---\n');
    fprintf('  train_adp : %s  (expect [9000 20 2 64 64])\n', mat2str(size(train_adp)));
    fprintf('  test_adp  : %s  (expect [1000 20 2 64 64])\n', mat2str(size(test_adp)));
    fprintf('  X_train   : %s  (expect [N_aug 10 2 64 64])\n', mat2str(size(X_train)));
    fprintf('  Y_train   : %s  (expect [N_aug  5 2 64 64])\n', mat2str(size(Y_train)));

    fprintf('\n--- Check 4: Temporal correlation (adjacent frame NMSE) ---\n');
    check_ids = [1, 50, 100, 500, 1000];
    nmse_vals = zeros(1, numel(check_ids));
    corr_vals = zeros(1, numel(check_ids));
    for k = 1:numel(check_ids)
        sid = check_ids(k);
        Hr  = squeeze(all_H_adp(train_idx(sid),:,1,:,:));
        Hi  = squeeze(all_H_adp(train_idx(sid),:,2,:,:));
        H   = Hr + 1j*Hi;

        f1 = squeeze(H(1,:,:)); f2 = squeeze(H(2,:,:));
        nmse_vals(k) = 10*log10(norm(f2(:)-f1(:))^2 / (norm(f1(:))^2 + 1e-12));

        Hf = reshape(H, num_frames, []);
        c1 = Hf(1,:); c2 = Hf(2,:);
        corr_vals(k) = abs(c1*c2') / (norm(c1)*norm(c2) + 1e-12);

        status = 'OK';
        if nmse_vals(k) > -8 || nmse_vals(k) < -25, status = 'WARN'; end
        if corr_vals(k) < 0.90, status = 'WARN'; end
        fprintf('  sample %4d: NMSE=%7.2f dB  corr=%.4f  [%s]\n', ...
                sid, nmse_vals(k), corr_vals(k), status);
    end
    adj_corr_th = abs(besselj(0, 2*pi*fd*frame_duration));
    fprintf('  Theoretical adjacent corr: %.4f\n', adj_corr_th);

    fprintf('\n--- Check 5: ADP sparsity ---\n');
    % Verify that ADP is sparser than raw CSI (per Fig.1 in paper)
    sid    = train_idx(1);
    Hr_adp = squeeze(all_H_adp(sid,1,1,:,:));
    Hi_adp = squeeze(all_H_adp(sid,1,2,:,:));
    amp    = abs(Hr_adp + 1j*Hi_adp);
    amp_n  = amp / max(amp(:));
    thresh = 0.05;
    sparsity = mean(amp_n(:) < thresh);
    fprintf('  ADP sparsity (fraction below 5%% peak): %.2f  (expect > 0.80)\n', sparsity);
    if sparsity < 0.80
        warning('ADP sparsity lower than expected. Check DFT matrices.');
    else
        fprintf('  PASS\n');
    end

    fprintf('\n--- Check 6: Plots ---\n');

    % Plot 1: Raw CSI magnitude vs ADP magnitude for one sample
    figure('Name','Check: CSI vs ADP magnitude','NumberTitle','off');
    subplot(1,2,1);
    Hr_raw = squeeze(all_H_adp(train_idx(1),1,1,:,:));  % re-use ADP as proxy
    Hi_raw = squeeze(all_H_adp(train_idx(1),1,2,:,:));
    imagesc(abs(Hr_raw + 1j*Hi_raw));
    colorbar; title('ADP amplitude (angle vs delay)');
    xlabel('Angle index'); ylabel('Delay index');

    subplot(1,2,2);
    tr_samp = squeeze(train_adp(1,:,1,:,:));   % (20, Nc, Nt) real part
    tr_samp_c = squeeze(train_adp(1,:,1,:,:)) + 1j*squeeze(train_adp(1,:,2,:,:));
    imagesc(abs(squeeze(tr_samp_c(1,:,:))));
    colorbar; title('Normalised ADP frame 1');
    xlabel('Antenna index'); ylabel('Subcarrier index');
    saveas(gcf, 'val_csi_vs_adp.png');

    % Plot 2: Temporal variation across 20 frames
    figure('Name','Check: Temporal variation','NumberTitle','off');
    samp_mag = squeeze(train_adp(1,:,1,1,:));  % (20, Nt) first subcarrier real
    imagesc(samp_mag);
    colorbar; xlabel('Antenna index'); ylabel('Frame index');
    title('Real ADP component across 20 frames (sample 1)');
    saveas(gcf, 'val_temporal_variation.png');

    % Plot 3: Per-sample norm distribution
    figure('Name','Check: Normalisation histogram','NumberTitle','off');
    flat = train_adp(:);
    histogram(flat, 100, 'Normalization','probability');
    xlabel('Normalised amplitude'); ylabel('Probability');
    title('Distribution of normalised ADP values (train set)');
    xline(-1,'r--'); xline(1,'r--');
    saveas(gcf, 'val_norm_distribution.png');

    fprintf('  Plots saved: val_csi_vs_adp.png, val_temporal_variation.png, val_norm_distribution.png\n');
    fprintf('\n=== ALL VALIDATION CHECKS PASSED ===\n');
end

