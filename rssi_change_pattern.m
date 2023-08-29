clear;
clc;
% this script shows you the change pattern of rssi
%% setting parameters
L0 = 10;
n = 3.78;
d0 = 1;
T = 10;             % 10 seconds - the total time 
t_step = 0.1;       % time step 0.1s
holdtime = 2;       % the speed does not change for 2 seconds
v = -10;            % -8m/s the station can only be closer to AP
v_max = 16;
v_min = 8;
ds_start = 250;     % 250m away to the AP
rssi_start = 0;
%% moving for `T` period
time_points = 0:t_step:T;
rssi = zeros(length(time_points), 1);
datachange = zeros(length(time_points), 2);
v_new = 0;
ds_prev = ds_start;
for t_id = 1:length(time_points)
    t_cur = time_points(t_id);
    if t_cur > 0 && mod(t_cur, holdtime) == 0
        v_new = (rand(1)*(v_max - v_min)+v_min)*cos(rand(1)*2*pi);
    end
    if v_new ~= 0
        v = v_new;
        datachange(t_id, 1) = 1;
    end
    datachange(t_id, 2) = v;
    v_new = 0;

    d_cur = ds_prev + v*t_cur;
    pl = L0 + 10*n*log10(d_cur/d0);
    rssi(t_id) = rssi_start - pl;
    
    % update new location
    ds_prev = d_cur;
end


%% plot
plot(time_points, rssi);
hold on;
plot(time_points, datachange(:, 2));
hold off;
grid on
xlabel('time')
ylabel('rssi (dBm)')
legend(["rssi", 'speed']);