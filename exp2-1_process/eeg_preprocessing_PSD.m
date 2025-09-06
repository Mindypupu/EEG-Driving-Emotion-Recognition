% 加載 EEGLAB（以非 GUI 模式）
clear;
eeglab nogui;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%  需更改
% 定義路徑
input_path = 'C:\codehome\eegdata\input';


%%%%%%%%%%%%%%%%%%%%%%%%%%%%  需更改
% 將所有result存到這個資料夾
output_path= 'C:\codehome\eegdata\output';
% 將所有前處理完的資料另外儲存


% 若檔案是 .edf 檔案
% 獲取所有 .edf文件的列表
edf_files = dir(fullfile(input_path, '*.edf'));
%edf_files = dir(fullfile(input_path, '*.cdt'));
file_list = {edf_files.name};


% 若檔案是 .set 檔案
% 獲取所有 .set文件的列表
% edf_files = dir(fullfile(input_path, '*.set'));
% file_list = {edf_files.name};

%%%%%%%%%%%%%%%%%%%%%%%%%%%%  需更改
% 定義通道名稱
channel_labels = {'Fp1', 'Fp2', 'F7', 'F3', 'FZ', 'F4', 'F8', 'FT7', 'FC3'...
                  'FCZ','FC4', 'FT8', 'T3', 'C3', 'CZ', 'C4', 'T4', 'TP7'...
                  'CP3', 'CPZ', 'CP4', 'TP8', 'A1', 'T5', 'P3', 'PZ', 'P4' ...
                  'T6', 'A2', 'O1', 'OZ', 'O2', 'HEOL', 'HEOR-L', 'VEOU'...
                  'VEOL-U', 'TRIGGER'};

%%%%%%%%%%%%%%%%%%%%%%%%%%%%  需更改
% 定義通道索引
%frontal_indices = [1, 2, 3, 4, 5, 6, 18];
%temporal_left_indices = [4, 6, 8, 10, 12, 14];
%temporal_right_indices = [3, 5, 7, 9, 11, 13];
%parietal_indices = [11, 12, 17, 18, 19];
%occipital_indices = [9, 10, 13, 14, 15, 16, 19];
%all_indices = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19];

frontal_indices = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11];
temporal_left_indices = [7, 12, 17, 23];
temporal_right_indices = [11, 16, 21, 27];
parietal_indices = [18, 19, 20, 21, 22, 24, 25, 26, 27, 28];
occipital_indices = [30, 31, 32];
all_indices = 1:32;  % 排除眼電通道

% 定義功率頻段
delta_range = [1 4];
theta_range = [5 7];
alpha_range = [8 12];
beta_range = [13 28];
gamma_range = [30 50];
all_range = [1 50];

% 處理每個文件
for file_idx = 1:length(file_list)
    % 加載文件並轉換為 EEGLAB 格式
     edf_file = fullfile(input_path, file_list{file_idx});
    %EEG = pop_loadset(edf_file); % 若檔案是 .set檔案
     EEG = pop_biosig(edf_file);  % 若檔案室 .edf檔案
     %EEG = pop_loadcurry(edf_file);
    
    EEG = pop_select(EEG, 'nochannel', {'HEOL', 'HEOR-L', 'VEOU', 'VEOL-U', 'VEOL-U', 'TRIGGER'});
    
    % 設置通道名稱
    num_channels = EEG.nbchan;
    channel_labels = {'Fp1', 'Fp2', 'F7', 'F3', 'FZ', 'F4', 'F8', 'FT7', 'FC3'...
                  'FCZ','FC4', 'FT8', 'T3', 'C3', 'CZ', 'C4', 'T4', 'TP7'...
                  'CP3', 'CPZ', 'CP4', 'TP8', 'A1', 'T5', 'P3', 'PZ', 'P4' ...
                  'T6', 'A2', 'O1', 'OZ', 'O2'};
    for i = 1:num_channels
        EEG.chanlocs(i).labels = channel_labels{i};
    end

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%  需更改
    % 加載通道位置
    EEG = pop_chanedit(EEG, 'lookup', 'C:\Users\Mindy\OneDrive\桌面\eeglab\plugins\dipfit\standard_BEM\elec\standard_1005.elc');
    
    
    %頻率濾波
    EEG = pop_eegfiltnew(EEG, 'locutoff', 0.5, 'hicutoff', 50, 'plotfreqz', 0);

    %  figure; 
    % [spectopo_outputs,freqs]=pop_spectopo(EEG, 1, [], 'EEG' , 'freqrange',[0.5 50],'electrodes','off');

    % 執行 ICA
    EEG = pop_runica(EEG, 'icatype', 'runica', 'extended', 1, 'interrupt', 'on');

    


    % 標記並標旗 ICA 成分
    EEG = pop_iclabel(EEG, 'default');



    categories = {'Brain', 'Muscle', 'Eye', 'Heart', 'Line Noise', 'Channel Noise', 'Other'};
    labels_prob = EEG.etc.ic_classification.ICLabel.classifications;

    for ic = 1:size(labels_prob, 1)
        fprintf('Component %d probabilities:\n', ic);
        for c = 1:length(categories)
            fprintf('  %s: %.4f\n', categories{c}, labels_prob(ic, c));
        end
        fprintf('\n');
    end



    EEG = pop_icflag(EEG, [NaN NaN; 0.6 1; 0.6 1; 0.6 1; NaN NaN; NaN NaN; 0.6 1]);
    EEG = pop_subcomp(EEG, [], 0);                                  

    % 移除指定通道
    % EEG = pop_select(EEG, 'rmchannel', {'EKG'});

    % % 再次執行 ICA
    % EEG = pop_runica(EEG, 'icatype', 'runica', 'extended', 1, 'interrupt', 'on');
    % EEG = pop_iclabel(EEG, 'default');
    % EEG = pop_icflag(EEG, [NaN NaN; 0.6 1; 0.6 1; 0.6 1; NaN NaN; NaN NaN; 0.6 1]);
    % EEG = pop_subcomp(EEG, [], 0);

    % 更新數據集名稱
    [~, name, ~] = fileparts(file_list{file_idx});
    EEG.setname = name;
    EEG = eeg_checkset(EEG);

    % 儲存前處理後的數據集為 SET 文件
    %%output_preprocessed_filename_set = fullfile(preprocessed_path_1, [name, '_filter.set']);
    %%pop_saveset(EEG, 'filename', output_preprocessed_filename_set);

    % 儲存前處理後的數據集為 EDF 文件
    output_preprocessed_filename_edf = fullfile(output_path, [name, '_filter.edf']);
    pop_writeeeg(EEG, output_preprocessed_filename_edf, 'TYPE', 'EDF');


    % 計算功率譜
    figure; 
    [spectopo_outputs,freqs]=pop_spectopo(EEG, 1, [], 'EEG' , 'freqrange',[0.5 50],'electrodes','off');
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%畫圖

    % 創建對應的文件夾
    output_folder = fullfile(output_path, name);
    if ~exist(output_folder, 'dir')
        mkdir(output_folder);
    end

    % 保存圖像
    image_filename = fullfile(output_folder, [name, '_spectopo.png']);
    saveas(gcf, image_filename);
    close(gcf);  % 關閉圖形窗口

    % 計算各頻段功率
    delta_power = mean(spectopo_outputs(:, find(freqs >= delta_range(1) & freqs <= delta_range(2))), 2);
    theta_power = mean(spectopo_outputs(:, find(freqs >= theta_range(1) & freqs <= theta_range(2))), 2);
    alpha_power = mean(spectopo_outputs(:, find(freqs >= alpha_range(1) & freqs <= alpha_range(2))), 2);
    beta_power = mean(spectopo_outputs(:, find(freqs >= beta_range(1) & freqs <= beta_range(2))), 2);
    gamma_power = mean(spectopo_outputs(:, find(freqs >= gamma_range(1) & freqs <= gamma_range(2))), 2);
    all_power = mean(spectopo_outputs(:, find(freqs >= all_range(1) & freqs <= all_range(2))), 2);

    % 定義計算平均功率的函數
    compute = @(mean_power, indices) mean(mean_power(indices, 1));

    % 計算各腦區的平均功率
    delta_frontal = compute(delta_power, frontal_indices);
    theta_frontal = compute(theta_power, frontal_indices);
    alpha_frontal = compute(alpha_power, frontal_indices);
    beta_frontal = compute(beta_power, frontal_indices);
    gamma_frontal = compute(gamma_power, frontal_indices);
    all_frontal = compute(all_power, frontal_indices);

    delta_temporal_left = compute(delta_power, temporal_left_indices);
    theta_temporal_left = compute(theta_power, temporal_left_indices);
    alpha_temporal_left = compute(alpha_power, temporal_left_indices);
    beta_temporal_left = compute(beta_power, temporal_left_indices);
    gamma_temporal_left = compute(gamma_power, temporal_left_indices);
    all_temporal_left = compute(all_power, temporal_left_indices);

    delta_temporal_right = compute(delta_power, temporal_right_indices);
    theta_temporal_right = compute(theta_power, temporal_right_indices);
    alpha_temporal_right = compute(alpha_power, temporal_right_indices);
    beta_temporal_right = compute(beta_power, temporal_right_indices);
    gamma_temporal_right = compute(gamma_power, temporal_right_indices);
    all_temporal_right = compute(all_power, temporal_right_indices);

    delta_parietal = compute(delta_power, parietal_indices);
    theta_parietal = compute(theta_power, parietal_indices);
    alpha_parietal = compute(alpha_power, parietal_indices);
    beta_parietal = compute(beta_power, parietal_indices);
    gamma_parietal = compute(gamma_power, parietal_indices);
    all_parietal = compute(all_power, parietal_indices);

    delta_occipital = compute(delta_power, occipital_indices);
    theta_occipital = compute(theta_power, occipital_indices);
    alpha_occipital = compute(alpha_power, occipital_indices);
    beta_occipital = compute(beta_power, occipital_indices);
    gamma_occipital = compute(gamma_power, occipital_indices);
    all_occipital = compute(all_power, occipital_indices);

    delta_all = compute(delta_power, all_indices);
    theta_all = compute(theta_power, all_indices);
    alpha_all = compute(alpha_power, all_indices);
    beta_all = compute(beta_power, all_indices);
    gamma_all = compute(gamma_power, all_indices);
    all_all = compute(all_power, all_indices);

    result = [delta_frontal, theta_frontal, alpha_frontal, beta_frontal, gamma_frontal, all_frontal,...
              delta_temporal_left, theta_temporal_left, alpha_temporal_left, beta_temporal_left, gamma_temporal_left, all_temporal_left,...
              delta_temporal_right, theta_temporal_right, alpha_temporal_right, beta_temporal_right, gamma_temporal_right, all_temporal_right,...
              delta_parietal, theta_parietal, alpha_parietal, beta_parietal, gamma_parietal, all_parietal,...
              delta_occipital, theta_occipital, alpha_occipital, beta_occipital, gamma_occipital, all_occipital,...
              delta_all, theta_all, alpha_all, beta_all, gamma_all, all_all];

    raw_result = 10.^(result/10);
    % 替換空格為底線
    var_name = sprintf('result_%s', strrep(name, ' ', '_'));
    % 使用變數名存儲結果
    %results.(var_name) = raw_result;

    % 保存結果
    %save(fullfile(output_folder, [name, '_delta.mat']), 'delta_power');
    %save(fullfile(output_folder, [name, '_theta.mat']), 'theta_power');
    %save(fullfile(output_folder, [name, '_alpha.mat']), 'alpha_power');
    %save(fullfile(output_folder, [name, '_beta.mat']), 'beta_power');
    %save(fullfile(output_folder, [name, '_gamma.mat']), 'gamma_power');
    %save(fullfile(output_folder, [name, '_all.mat']), 'all_power');
    %save(fullfile(output_folder, [name, '_result.mat']), 'result');
    
end
