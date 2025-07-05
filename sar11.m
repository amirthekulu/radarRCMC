% MATLAB Script to Simulate and Process SAR Data for a Point Target using RDA

% --- Simulation Parameters ---
c = 3e8;                     % Speed of light (m/s)
fc = 5e9;                    % Carrier frequency (Hz) (e.g., C-band)
lambda = c/fc;               % Wavelength (m)
B = 100e6;                   % Chirp bandwidth (Hz)
Tp = 5e-6;                   % Pulse duration (s)
Kr = B/Tp;                   % Chirp rate (Hz/s)

R0 = 5000;                   % Closest slant range to target (m)
vp = 150;                    % Platform velocity (m/s)
Lsar = 500;                  % Synthetic aperture length (m)
Ta = Lsar / vp;              % Aperture time (s)
PRF = 500;                   % Pulse Repetition Frequency (Hz)
Na = round(Ta * PRF);        % Number of azimuth samples
if mod(Na,2) ~= 0
    Na = Na+1; % Ensure Na is even for easier FFT handling
end
ts = linspace(-Ta/2, Ta/2, Na); % Slow time vector (s)

% --- Fast Time (Range) Sampling Parameters ---
R_max_rcm = sqrt(R0^2 + (Lsar/2)^2); % Max range due to RCM
tr_start_time = 2*R0/c - Tp;         % Start time for earliest echo
tr_end_time   = 2*R_max_rcm/c + Tp;  % End time for latest echo
Nr = 2048; % Number of samples in fast time (range bins)
tr = linspace(tr_start_time, tr_end_time, Nr); % Fast time vector (s)
dt_fast = tr(2)-tr(1);             % Fast time sampling interval
fs_fast_actual = 1/dt_fast;        % Actual fast time sampling frequency
range_resolution = c/(2*B);
fprintf('Range resolution: %.2f m\n', range_resolution);
fprintf('RCM amount: %.2f m (approx %.2f range bins)\n', R_max_rcm-R0, (R_max_rcm-R0)/range_resolution);


% --- Simulate Raw Data ---
s_raw = zeros(Nr, Na); % Initialize raw data matrix
fprintf('Simulating raw data...\n');
for ia = 1:Na % Loop over azimuth positions
    ta_curr = ts(ia); % Current slow time
    R_inst = sqrt(R0^2 + (vp*ta_curr)^2); % Instantaneous slant range
    delay_inst = 2*R_inst/c; % Instantaneous round-trip delay

    % Generate received chirp
    pulse_signal_complex = exp(1j*pi*Kr*(tr - delay_inst - Tp/2).^2);
    rect_window = (tr >= delay_inst) & (tr <= (delay_inst + Tp));
    pulse_signal_complex = pulse_signal_complex .* rect_window;

    % Modulate with carrier phase (baseband conversion) and store
    s_raw(:,ia) = pulse_signal_complex.' .* exp(-1j*4*pi*fc*R_inst/c);
    if mod(ia, round(Na/10)) == 0
        fprintf('Processed %d/%d azimuth samples for raw data.\n', ia, Na);
    end
end
fprintf('Raw data simulation complete.\n');

% --- 1. Range Compression ---
fprintf('Performing Step 1: Range Compression...\n');
N_pulse_samples = ceil(Tp * fs_fast_actual);
t_ref_pulse = linspace(0, Tp, N_pulse_samples);
ref_chirp_time_domain = exp(1j*pi*Kr*(t_ref_pulse - Tp/2).^2);
ref_chirp_padded = zeros(Nr,1);
ref_chirp_padded(1:N_pulse_samples) = ref_chirp_time_domain;
H_matched_filter_freq = conj(fft(ref_chirp_padded, Nr)); % Matched filter in freq domain

S_raw_freq = fft(s_raw, Nr, 1); % FFT along range
S_range_compressed_freq = S_raw_freq .* repmat(H_matched_filter_freq, 1, Na); % Multiply by matched filter
s_range_compressed = ifft(S_range_compressed_freq, Nr, 1); % IFFT back to time domain
fprintf('Range compression complete.\n');

% Axes for plotting
azimuth_axis_m = ts * vp;
range_axis_m = tr * c/2;

% Define common Y-limits for the first three plots to see RCM clearly
ylim_center = (R0 + R_max_rcm)/2;
ylim_span = (R_max_rcm - R0) + 40*range_resolution; % Span of RCM + margin
ylim_plot_rcm = [ylim_center - ylim_span/2, ylim_center + ylim_span/2];

% --- Create Figure for Subplots ---
figure('Name', 'SAR RDA Processing Stages', 'NumberTitle', 'off', 'Position', [100, 100, 1200, 800]);

% Subplot 1: After Range Compression
subplot(2,2,1);
imagesc(azimuth_axis_m, range_axis_m, abs(s_range_compressed));
xlabel('Azimuth Position (m)');
ylabel('Range (m)');
title('1. After Range Compression (RCM Visible)');
colorbar;
colormap(gca, 'gray'); % Apply colormap to current axes
ylim(ylim_plot_rcm);
fprintf('Plotted image after range compression.\n');

% --- 2. Azimuth FFT (Transform to Range-Doppler Domain) ---
fprintf('Performing Step 2: Azimuth FFT...\n');
S_rd = fftshift(fft(s_range_compressed, Na, 2), 2); % FFT along azimuth, then shift
fprintf('Azimuth FFT complete.\n');

% Azimuth frequency axis
fa_axis = ((0:Na-1) - floor(Na/2)) * (PRF/Na); % Centered Doppler frequency axis

% Subplot 2: In Range-Doppler Domain (Before RCMC)
subplot(2,2,2);
imagesc(fa_axis, range_axis_m, abs(S_rd));
xlabel('Doppler Frequency (Hz)');
ylabel('Range (m)');
title('2. Data in Range-Doppler Domain (Before RCMC)');
colorbar;
colormap(gca, 'gray');
ylim(ylim_plot_rcm);
fprintf('Plotted image in Range-Doppler domain.\n');

% --- 3. Range Cell Migration Correction (RCMC) ---
fprintf('Performing Step 3: Range Cell Migration Correction (RCMC)...\n');
S_rcmc = zeros(Nr, Na, 'like', 1j); % Initialize complex matrix for RCMC output
range_grid_out_for_interp = range_axis_m.'; % Ensure it's a column vector for interp1

for k_fa = 1:Na % Loop over each Doppler frequency bin
    fa = fa_axis(k_fa); % Current Doppler frequency

    term_sq = (lambda * fa / (2*vp))^2;
    if term_sq >= 1
        S_rcmc(:, k_fa) = 0; % Set to zero for invalid/extreme Doppler frequencies
        continue;
    end

    delta_R_fa = R0 * (1/sqrt(1 - term_sq) - 1); % Range migration
    range_grid_in_for_interp = range_grid_out_for_interp + delta_R_fa; % Shifted range grid for interpolation

    % Interpolate S_rd at the shifted range positions
    S_rcmc(:, k_fa) = interp1(range_axis_m, S_rd(:, k_fa), range_grid_in_for_interp, 'linear', 0);
end
fprintf('RCMC complete.\n');

% Subplot 3: After RCMC (in Range-Doppler Domain)
subplot(2,2,3);
imagesc(fa_axis, range_axis_m, abs(S_rcmc));
xlabel('Doppler Frequency (Hz)');
ylabel('Range (m)');
title('3. Data in Range-Doppler Domain (After RCMC)');
colorbar;
colormap(gca, 'gray');
ylim(ylim_plot_rcm);
fprintf('Plotted image after RCMC.\n');

% --- 4. Azimuth Compression ---
fprintf('Performing Step 4: Azimuth Compression...\n');
H_az_matched = zeros(1, Na, 'like', 1j); % Row vector for azimuth matched filter
for k_fa = 1:Na
    fa = fa_axis(k_fa);
    term_sq = (lambda * fa / (2*vp))^2;
    if term_sq < 1 % Only apply filter for valid Doppler frequencies
        H_az_matched(k_fa) = exp(1j * (4*pi*R0/lambda) * sqrt(1 - term_sq));
    else
        H_az_matched(k_fa) = 0; % No signal contribution expected here
    end
end

S_az_compressed_rd = S_rcmc .* repmat(H_az_matched, Nr, 1); % Apply filter
fprintf('Azimuth compression complete.\n');

% --- 5. Azimuth IFFT (Transform back to Image Domain) ---
fprintf('Performing Step 5: Azimuth IFFT...\n');
s_focused = ifft(ifftshift(S_az_compressed_rd, 2), Na, 2); % Inverse shift, then IFFT
fprintf('Azimuth IFFT complete.\n');

% Subplot 4: Final Focused Image
subplot(2,2,4);
imagesc(azimuth_axis_m, range_axis_m, abs(s_focused));
xlabel('Azimuth Position (m)');
ylabel('Range (m)');
title('4. Final Focused Image');
colorbar;
colormap(gca, 'gray');

% Zoom into the target area for the final image
ylim_focused = [R0 - 20*range_resolution, R0 + 20*range_resolution];
xlim_focused = [-Lsar/10, Lsar/10]; % Show a smaller azimuth extent around target
ylim(ylim_focused);
xlim(xlim_focused);
fprintf('Plotted final focused image.\n');

sgtitle('SAR RDA Processing Stages for a Point Target', 'FontSize', 16, 'FontWeight', 'bold'); % Super title for the whole figure

fprintf('RDA processing and plotting finished.\n');
% --- End of Script ---
