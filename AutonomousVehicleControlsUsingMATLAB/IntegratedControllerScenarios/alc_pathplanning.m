function path = alc_pathplanning(V,Tf,Ts,alc_dir)
% usage = alc_pathplannig(V,Tf,Ts)
%% calculate polynomial coefficients
V = V;
L = 3.5; % lane width

T0 = 0;
Tf = Tf;
x_f = V*Tf;
t = 0:Ts:Tf;
T = [1 T0 T0^2 T0^3 T0^4 T0^5;
     0  1  2*(T0) 3*(T0)^2  4*(T0)^3 5*(T0)^4;
     0 0 2 6*(T0) 12*(T0)^2 20*(T0)^3;
     1 Tf Tf^2 Tf^3 Tf^4 Tf^5;
     0 1 2*Tf 3*(Tf)^2 4*(Tf)^3 5*(Tf)^4;
     
     0 0 2 6*Tf 12*(Tf)^2 20*(Tf)^3];
 
Q_x = [0 ; V; 0; x_f; V; 0];
Q_y = [0;0;0; L; 0;0];

a = inv(T)*Q_x;
b = inv(T)*Q_y;

X_ref = a(1) + a(2)*t + a(3)*t.^2 + a(4)*t.^3 + a(5)*t.^4 + a(6)*t.^5;
Y_ref = b(1) + b(2)*t + b(3)*t.^2 + b(4)*t.^3 + b(5)*t.^4 + b(6)*t.^5;

Y_ref = alc_dir*Y_ref; % lateral positions wil based on the direction of turn indicator


Vx = a(2) + 2*a(3)*t + 3*a(4)*t.^2 +4*a(5)*t.^3+ a(6)*t.^4;
Vy = b(2) + 2*b(3)*t + 3*b(4)*t.^2 +4*b(5)*t.^3+ b(6)*t.^4;

ay = 2*b(3) + 6*b(4)*t + 12*b(5)*t.^2 + 20*b(6)*t.^3;
a_max = max(abs(ay));
%% check for lateral accleration
n = 10
for ii = 1:n
    if a_max>2
        Tf = Tf+1;
        x_f = V*Tf;
        t = 0:0.1:Tf;
        T = [1 T0 T0^2 T0^3 T0^4 T0^5;
             0  1  2*(T0) 3*(T0)^2  4*(T0)^3 5*(T0)^4;
             0 0 2 6*(T0) 12*(T0)^2 20*(T0)^3;
             1 Tf Tf^2 Tf^3 Tf^4 Tf^5;
             0 1 2*Tf 3*(Tf)^2 4*(Tf)^3 5*(Tf)^4;

             0 0 2 6*Tf 12*(Tf)^2 20*(Tf)^3];

        Q_x = [0 ; V; 0; x_f; V; 0]
        Q_y = [0;0;0; L; 0;0]

        a = inv(T)*Q_x;
        b = inv(T)*Q_y;

        X_ref = a(1) + a(2)*t + a(3)*t.^2 + a(4)*t.^3 + a(5)*t.^4 + a(6)*t.^5;
        Y_ref = b(1) + b(2)*t + b(3)*t.^2 + b(4)*t.^3 + b(5)*t.^4 + b(6)*t.^5;

        Vx = a(2) + 2*a(3)*t + 3*a(4)*t.^2 +4*a(5)*t.^3+ a(6)*t.^4;
        Vy = b(2) + 2*b(3)*t + 3*b(4)*t.^2 +4*b(5)*t.^3+ b(6)*t.^4;

        ay = 2*b(3) + 6*b(4)*t + 12*b(5)*t.^2 + 20*b(6)*t.^3;
        a_max = max(abs(ay));
    else
        break
    end
end

    
%% calculate yaw
xRef = X_ref;
yRef = Y_ref;

for ii = 2:length(xRef) - 1
    x_forward = xRef(ii+1);
    x_back = xRef(ii-1);
    y_forward = yRef(ii+1);
    y_back = yRef(ii-1);
    yawRef(ii) = atan2(y_forward -y_back,  x_forward-x_back);
end
yawRef(1) = yawRef(2);
yawRef(end) = yawRef(end-1);
yawRef(1,length(yawRef)+1) = yawRef(end);

path.xRef = xRef;
path.yRef = yRef;
path.yawRef = yawRef;
path.T_lane = Tf;






 

