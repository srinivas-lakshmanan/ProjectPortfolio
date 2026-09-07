% initalization script for feature model for Lance centering assist
%%
clc;
clear all; close all
%% Waypoints extraction from scenarios

% scenario = course1_lancechange;
%scenario = lane_change_course_1;
scenario = straight_road;
n = size(scenario);
Vx = 15; %mps
V_initial = 15; %mps
for ii =1:n(2)
    refPose(:,ii) = [scenario(ii).ActorPoses.Position(1);scenario(ii).ActorPoses.Position(2);scenario(ii).ActorPoses.Yaw];
    yaw_rate_pose(ii) = scenario(ii).ActorPoses.AngularVelocity(3);
end 
xRef = refPose(1,:);
yRef = refPose(2,:) ;
% yRef = refPose(2,:) + 2; %for lane change course
yawRef = refPose(3,:);
refPose = refPose'; % reference waypoints
yaw_rate_pose = deg2rad(yaw_rate_pose); %angular velocity of the path

%% define reference time 
sim_time = 50; % simulation time for city road

s = size(xRef);
tRef = (linspace(0,sim_time,s(2)))'; % this time variable is used in the "2D Visualization" block for plotting the reference points. 


%%
% Define the sample time, |Ts|, and simulation duration, |T|, in seconds.
Ts = 0.1; % sample time for MPC

%% curvature preview and calculation
PredictionHorizon = 10;
md =get_curvature(Vx,yaw_rate_pose,tRef');
% data = mpc_previewer_maskinit(md,tRef');
u_min = -0.5;
u_max = 0.5;
Radius =abs(1./ md.signals.values);
%% Model parameters
m = 1140;
I_z = 1436.24; %kgm2
l_f = 1.165; 
l_r = 1.165;
c_f = 155494.663;
c_r = 1554494.663;

m = 1140;
Iz = 1436.24;
lf = 1.165;
lr = 1.165;
Cf = 155494.663;
Cr =1554494.663;

% assuming constant velocity
v_x = Vx; %mps
%% Dynamic bicycle model 
% continous A B C D matrixes
A = [0 1 Vx 0;
     0 -2*(c_f+c_r)/(m*v_x) 0 -2*(l_f*c_f + l_r*c_r)/(m*v_x) - v_x;
     0 0 0 1;
     0 2*(l_r*c_r - l_f*c_f)/(I_z*v_x)  0  -2*(l_r^2*c_r - l_f^2*c_f)/(I_z*v_x)];
 
B = [0;
    2*c_f/m;
    0;
    2*l_f*c_f/I_z];

C= [1 0 0 0;
       0 1 0 0];
D = [0;0];
%%
vehicle = ss(A,B,C,D)

%% Parameters for Pure Pursuit
L = 4.7; % bicycle length
ld = 10; % lookahead distance
X_o = xRef(1,1); % initial vehicle position
Y_o = yRef(1,1); % initial vehicle position 
psi_o = 0; % for lane change
% psi_o =   -1.617; % initial yaw angle
% psi_o =4.17239; % initial yaw angle for fig eight



%% velocity and time 
Tf = 5; % time to complete lane change
Ts= 0.1;
alc_dir = -1; % R to L =+1  , L to R = -1
Vx = 15; %mps
Tf = 5; % time to complete lane change
Ts= 0.1;
alc_dir = 1; % R to L =+1  , L to R = -1
tRef_alc=0;
xRef_alc = 0;
yRef_alc = 0;
yawRef_alc = 0;

%%

load('mpc1.mat');


%%
mdl = 'integrated_controller_scenario_q6_MRM';
open_system(mdl)
time = tic;

sim(mdl)

%% velocity and time 
Vx = 15; %mps
Tf = 5; % time to complete lane change
Ts= 0.1;
alc_dir = 1; % R to L =+1  , L to R = -1
tRef_alc=0;
xRef_alc = 0;
yRef_alc = 0;
yawRef_alc = 0;
out.MRM_trigger = 0;
out.shoulder_direc = 0;


%% lane change path planning
if(out.MRM_trigger==1)
alc_dir = out.shoulder_direc
path = alc_pathplanning(Vx,Tf,Ts,alc_dir);
xRef_alc = path.xRef;
yRef_alc= path.yRef;
yawRef_alc = path.yawRef; 
T_lane = path.T_lane
tRef_alc = 0:Ts:T_lane;
xRef_max = max(xRef);
Vx = 0;
end

% yawRef(1,length(yawRef)+1) = yawRef(end);







%bdclose(mdl)


        


