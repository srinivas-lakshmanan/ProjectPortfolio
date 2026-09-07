clc;clear all;close all;
%% model parameters
m = 1084 ; % kg
g = 9.81 ; % mpss
rho = 1.2; % km/m3
Cd = 0.3;
f = 0.0125;
v_0 = 22; %mps
vw_0 = 1.2; %mps
A = 2.24; %mps2

%% plant model

plant_numerator = 9.221*1e-4;
plant_denominator = [1 1/46.353];
G_plant = tf(plant_numerator,plant_denominator);

%% PID gains
Kp= 490.9572;
Ki = 51.1361;
Kd = 0; 

%% ask user for speed limit

% prompt = {'enter the speed limit distance'}; dlgtitle = 'Input' ; dims =
% 1; definput = {'0'}; answer =
% cell2mat(inputdlg(prompt,dlgtitle,dims,definput));
% v_set_speedlimt_distance = str2double(answer) vset = 0; if
% v_set_speedlimt_distance == 100
%     vset = 25;
% elseif v_set_speedlimt_distance == 250
%     vset = 30;
% elseif v_set_speedlimt_distance == 500
%     vset = 45;
% elseif v_set_speedlimt_distance == 700
%     vset = 50;
% elseif v_set_speedlimt_distance == 1000
%     vset = 80;
% end


%% 


simTime = 1800; %sec (best if multiple of 9)
t = 0:1:simTime-1;


%% Generate signals for Red, Yellow and Green lights
% Entirecycle for R-G-Y consists of 90sec 40 +40 +10
% make 1 cycle with amplitude 1 for 90 sec/ elements. 
% first assume cycle starts with Red light hence first 40sec value is 1 and
% next 50 sec value is 0

Red_period = 1*[ones(1,40), zeros(1,50)];
Green_period = 1*[zeros(1,40), ones(1,40), zeros(1,10)];
Yellow_period = 1*[zeros(1,80), ones(1,10)];
% define number of cycles
numCycles = floor(simTime / numel(Red_period));
red_fullwave = repmat(Red_period, [1,numCycles]);
green_fullwave = repmat(Green_period, [1,numCycles]);
yellow_fullwave = repmat(Yellow_period, [1,numCycles]);

% One signal for all the traffic lights in a sequence of red-green-yellow
 traffic_light_color = [red_fullwave ; green_fullwave ; yellow_fullwave];
% plot the signal
plot(t(1:numel(red_fullwave)), red_fullwave, 'r-','LineWidth',2);
hold on;
plot(t(1:numel(yellow_fullwave)), yellow_fullwave, 'y-','LineWidth',2);
plot(t(1:numel(green_fullwave)), green_fullwave, 'g-','LineWidth',2);

fontSize =20;
xlabel('time (sec)' , 'FontSize',fontSize);
ylabel('Amplitude' , 'FontSize',fontSize); grid on;


%% phase time signal
% calculate what is the current Light value and h
% how much remaining time left in the current light

% light = Red = 1
% light = Green = 2
% light = yellow = 3
light= [];

for ii = 1:length(t)-1
    if traffic_light_color(1,ii) == 1
        % light is now red
        light(ii) = 1;
        time_elapsed_inphase = mod(t(ii),10); % how much time elased in this phase
        time_remain_inphase(ii) = 40-time_elapsed_inphase; % how much time left in this phase
    elseif traffic_light_color(2,ii) ==1
        % light is now green
        time_elapsed_inphase = mod(t(ii),10); % how much time elased in this phase
        time_remain_inphase(ii) = 40-time_elapsed_inphase; % how much time left in this phase
        light(ii) =2;
    else
        % light is now yellow
        time_elapsed_inphase = mod(t(ii),10); % how much time elased in this phase
        time_remain_inphase(ii) = 10-time_elapsed_inphase; % how much time left in this phase
        light(ii) =3;
        
    end
    
end
v2x_traffic_signal = [light;time_remain_inphase]'; % input to the vehicle DSRC from the v2x system

time_left_for_next_yellow_end = 1:90;
time_left_for_next_yellow_end = fliplr(time_left_for_next_yellow_end);

time_fullwave = repmat(time_left_for_next_yellow_end, [1,numCycles]);
v2x_traffic_signal = [light;time_fullwave(1:length(t)-1)]';
%% calculate vehicle distance from traffic light
% DSRC will only catch a signal at 400m
% assume constant velocity of the vehicle 15m/s
host_v0 = 15; %m/s
x0_host = 0; %m
for ii = 1:length(t)-1
    X_host(ii) = x0_host+ host_v0*t(ii);
end


%% Calculate Green wave velocity
% greenwave velocity will be a lower velocity such that the vehicle can
% travel within the greenwave time period.

% calculate green-wave time
for ii = 1:length(t)-1
    if v2x_traffic_signal(ii,1) == 1
        % light is red
    end
end

    
    